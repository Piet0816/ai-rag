package com.pseidl.ai.rag.web;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pseidl.ai.rag.config.AppConfig;
import com.pseidl.ai.rag.index.InMemoryVectorIndex;
import com.pseidl.ai.rag.ollama.EmbeddingClient;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.ArraySchema;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.tags.Tag;

/**
 * Retrieval-augmented chat. - /api/chat : classic one-shot JSON reply -
 * /api/chat/stream : SSE streaming reply (events: meta, delta, done, error)
 */
@RestController
@RequestMapping("/api/chat")
@Tag(name = "Chat", description = "Chat with your model augmented by the indexed library context")
public class ChatController {

	private final RestClient http;
	private final EmbeddingClient embeddings;
	private final InMemoryVectorIndex index;
	private final String chatModel;
	private final String baseUrl;
	private final ObjectMapper mapper = new ObjectMapper();

	@Value("${app.retrieval.top-k:6}")
	private int defaultTopK;

	@Value("${app.retrieval.max-context-chars:6000}")
	private int maxContextChars;

	public ChatController(RestClient.Builder builder, AppConfig app, EmbeddingClient embeddings,
			InMemoryVectorIndex index) {
		this.http = builder.baseUrl(app.getOllama().getBaseUrl()).build();
		this.baseUrl = app.getOllama().getBaseUrl();
		this.chatModel = app.getOllama().getChatModel();
		this.embeddings = embeddings;
		this.index = index;
	}

	// -------- non-streaming --------
	@PostMapping(value = "", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
	@Operation(summary = "Chat with retrieval", description = "Embeds the latest user message, retrieves top-K, builds context, and calls Ollama.")
	public ResponseEntity<ChatResponse> chat(@RequestBody ChatRequest req,
			@RequestParam(name = "topK", required = false) Integer topKParam,
			@RequestParam(name = "debug", defaultValue = "false") boolean debug) {
		try {
			Prepared p = prepare(req, topKParam);
			if (p.error != null)
				return ResponseEntity.badRequest().body(ChatResponse.error(p.error));

			String reply;
			try {
				Map<String, Object> body = new HashMap<>();
				body.put("model", chatModel);
				body.put("messages", p.messages);
				body.put("stream", false);

				JsonNode node = http.post().uri("/api/chat").contentType(MediaType.APPLICATION_JSON).body(body)
						.retrieve().body(JsonNode.class);

				reply = node != null && node.has("message") && node.get("message").has("content")
						? node.get("message").get("content").asText()
						: node != null && node.has("response") ? node.get("response").asText() : "[no content]";
			} catch (RestClientResponseException e) {
				// Fallback to /api/generate
				try {
					Map<String, Object> body = new HashMap<>();
					body.put("model", chatModel);
					body.put("prompt", p.latestUserText);
					body.put("system", p.system); // <-- fixed
					body.put("stream", false);

					JsonNode node = http.post().uri("/api/generate").contentType(MediaType.APPLICATION_JSON).body(body)
							.retrieve().body(JsonNode.class);

					reply = node != null && node.has("response") ? node.get("response").asText() : "[no content]";
				} catch (Exception ex) {
					return ResponseEntity.internalServerError()
							.body(ChatResponse.error("Chat request failed: " + ex.getMessage()));
				}
			}

			return ResponseEntity.ok(new ChatResponse(chatModel, reply, p.retrieved, debug ? p.context : null));
		} catch (Exception e) {
			return ResponseEntity.internalServerError().body(ChatResponse.error("Chat failed: " + e.getMessage()));
		}
	}

	// -------- streaming (SSE) --------
	@PostMapping(value = "/stream", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.TEXT_EVENT_STREAM_VALUE)
	@Operation(summary = "Chat with streaming + retrieval (SSE)", description = "Streams partial tokens as 'delta' events; emits 'meta' first with retrieved hits.")
	public SseEmitter chatStream(@RequestBody ChatRequest req,
			@RequestParam(name = "topK", required = false) Integer topKParam,
			@RequestParam(name = "debug", defaultValue = "false") boolean debug) {
		final SseEmitter emitter = new SseEmitter(0L);

		new Thread(() -> {
			try {
				Prepared p = prepare(req, topKParam);
				if (p.error != null) {
					sendEvent(emitter, "error", p.error);
					emitter.complete();
					return;
				}

				// meta first
				Map<String, Object> meta = new LinkedHashMap<>();
				meta.put("model", chatModel);
				meta.put("retrieved", p.retrieved);
				if (debug)
					meta.put("context", p.context);
				sendEvent(emitter, "meta", mapper.writeValueAsString(meta));

				// stream from Ollama
				URL url = new URL(baseUrl + "/api/chat");
				HttpURLConnection con = (HttpURLConnection) url.openConnection();
				con.setRequestMethod("POST");
				con.setDoOutput(true);
				con.setRequestProperty("Content-Type", "application/json; charset=utf-8");

				Map<String, Object> body = new HashMap<>();
				body.put("model", chatModel);
				body.put("messages", p.messages);
				body.put("stream", true);

				try (OutputStream os = con.getOutputStream()) {
					os.write(mapper.writeValueAsBytes(body));
					os.flush();
				}

				int code = con.getResponseCode();
				if (code < 200 || code >= 300) {
					try (BufferedReader er = new BufferedReader(
							new InputStreamReader(con.getErrorStream(), StandardCharsets.UTF_8))) {
						StringBuilder sb = new StringBuilder();
						String ln;
						while ((ln = er.readLine()) != null)
							sb.append(ln);
						sendEvent(emitter, "error", sb.toString());
					}
					emitter.complete();
					return;
				}

				try (BufferedReader br = new BufferedReader(
						new InputStreamReader(con.getInputStream(), StandardCharsets.UTF_8))) {
					String line;
					while ((line = br.readLine()) != null) {
						line = line.trim();
						if (line.isEmpty())
							continue;

						JsonNode node = mapper.readTree(line);
						String delta = null;
						if (node.has("message") && node.get("message").has("content")) {
							delta = node.get("message").get("content").asText();
						} else if (node.has("response")) {
							delta = node.get("response").asText();
						}
						if (delta != null && !delta.isEmpty())
							sendEvent(emitter, "delta", delta);
						if (node.has("done") && node.get("done").asBoolean(false))
							break;
					}
				}

				sendEvent(emitter, "done", "ok");
				emitter.complete();

			} catch (Exception e) {
				try {
					sendEvent(emitter, "error", e.getMessage());
				} catch (Exception ignore) {
				}
				emitter.completeWithError(e);
			}
		}, "chat-stream-worker").start();

		return emitter;
	}

	// -------- shared prep --------
	private Prepared prepare(ChatRequest req, Integer topKParam) {
		if (req == null || req.messages() == null || req.messages().isEmpty()) {
			return Prepared.error("Missing 'messages'");
		}
		if (!StringUtils.hasText(chatModel)) {
			return Prepared.error("Chat model is not configured (app.ollama.chat-model)");
		}

		ChatMessage latestUser = null;
		for (int i = req.messages().size() - 1; i >= 0; i--) {
			if ("user".equalsIgnoreCase(req.messages().get(i).role())) {
				latestUser = req.messages().get(i);
				break;
			}
		}
		if (latestUser == null || !StringUtils.hasText(latestUser.content())) {
			return Prepared.error("No user message found in 'messages'");
		}

		int topK = (topKParam != null && topKParam > 0) ? topKParam : Math.max(1, defaultTopK);

		float[] qvec = embeddings.embed(latestUser.content());
		var hits = index.search(qvec, topK);

		String context = buildContext(hits, maxContextChars);
		List<RetrievedHit> retrieved = hits.stream().map(h -> new RetrievedHit(h.source(), h.chunkIndex(), h.score()))
				.collect(Collectors.toList());

		String system = """
				You are a helpful assistant. Use the provided CONTEXT to answer the user's question.
				If the answer is not clearly in the context, say you don't know.
				Be concise and cite the source chunk like [source#chunk] when useful (e.g., [people_food.txt#0]).
				CONTEXT:
				""" + context;

		List<Map<String, String>> messages = new ArrayList<>();
		messages.add(Map.of("role", "system", "content", system));
		for (ChatMessage m : req.messages()) {
			if (!"system".equalsIgnoreCase(m.role()))
				messages.add(Map.of("role", m.role(), "content", m.content()));
		}

		return Prepared.ok(messages, retrieved, context, latestUser.content(), system);
	}

	private static void sendEvent(SseEmitter emitter, String name, String data) throws Exception {
		emitter.send(SseEmitter.event().name(name).data(data));
	}

	private static String buildContext(List<InMemoryVectorIndex.Result> hits, int maxChars) {
		if (hits == null || hits.isEmpty())
			return "(no matches)";
		StringBuilder sb = new StringBuilder(Math.min(maxChars, 8192));
		for (var h : hits) {
			String header = "\n[" + h.source() + "#" + h.chunkIndex() + "] (score " + String.format("%.3f", h.score())
					+ ")\n";
			if (sb.length() + header.length() > maxChars)
				break;
			sb.append(header);
			String txt = h.text();
			int remain = maxChars - sb.length();
			if (txt.length() <= remain)
				sb.append(txt);
			else {
				sb.append(txt, 0, Math.max(0, remain));
				break;
			}
		}
		return sb.toString();
	}

	// -------- DTOs --------
	public record ChatMessage(@Schema(description = "Role: user|assistant|system", example = "user") String role,
			@Schema(description = "Message content", example = "What does Alice like to eat?") String content) {
	}

	public record ChatRequest(
			@ArraySchema(schema = @Schema(implementation = ChatMessage.class)) List<ChatMessage> messages) {
	}

	public record RetrievedHit(@Schema(description = "Source file") String source,
			@Schema(description = "Chunk index") int chunkIndex, @Schema(description = "Cosine score") double score) {
	}

	public record ChatResponse(@Schema(description = "Model used") String model,
			@Schema(description = "Assistant reply") String answer,
			@ArraySchema(schema = @Schema(implementation = RetrievedHit.class)) List<RetrievedHit> retrieved,
			@Schema(description = "Raw context sent to the model (only when debug=true)") String context) {
		static ChatResponse error(String msg) {
			return new ChatResponse(null, msg, List.of(), null);
		}
	}

	// internal holder for shared prep
	private static final class Prepared {
		final List<Map<String, String>> messages;
		final List<RetrievedHit> retrieved;
		final String context;
		final String latestUserText;
		final String system; // <-- added
		final String error;

		private Prepared(List<Map<String, String>> messages, List<RetrievedHit> retrieved, String context,
				String latestUserText, String system, String error) {
			this.messages = messages;
			this.retrieved = retrieved;
			this.context = context;
			this.latestUserText = latestUserText;
			this.system = system;
			this.error = error;
		}

		static Prepared ok(List<Map<String, String>> messages, List<RetrievedHit> retrieved, String context,
				String latestUserText, String system) {
			return new Prepared(messages, retrieved, context, latestUserText, system, null);
		}

		static Prepared error(String msg) {
			return new Prepared(null, null, null, null, null, msg);
		}
	}
}
