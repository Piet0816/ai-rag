package com.pseidl.ai.rag.retrieval;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pseidl.ai.rag.config.AppConfig;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Asks a local Ollama model to extract short entity hints from a user prompt.
 * Returns a small, deduplicated list like ["Pikachu","Blastoise","Dragon"].
 *
 * Uses /api/generate (works for chat or reasoning models).
 */
@Service
public class OllamaEntityHintService {

    private final RestClient http;
    private final String model;
    private final ObjectMapper mapper = new ObjectMapper();

    /** Optional override: if empty, falls back to app.ollama.chat-model. */
    @Value("${app.ollama.hints-model:}")
    private String hintsModel;

    /** Safety cap to avoid very long lists from the model. */
    @Value("${app.retrieval.hints.max:6}")
    private int maxDefault;

    public OllamaEntityHintService(RestClient.Builder builder, AppConfig app) {
        this.http = builder.baseUrl(app.getOllama().getBaseUrl()).build();
        this.model = app.getOllama().getChatModel();
    }

    /**
     * Extract up to maxEntities entity hints using a strict JSON-only prompt.
     */
    public List<String> extractHints(String userText, int maxEntities) {
        if (userText == null || userText.isBlank()) return List.of();
        int cap = Math.max(1, maxEntities > 0 ? maxEntities : maxDefault);

        String sys = """
                Extract concise entity hints from the USER text.
                - Include proper names (people, products, characters), AND domain keywords useful for retrieval.
                - Do not include stopwords or filler.
                - Return ONLY a compact JSON array of strings on a single line, no prose, no code fences.
                Example: ["EntityA","CountryB","PersonC"]
                """;

        String prompt = "USER: " + userText + "\n" +
                        "Now return a JSON array with at most " + cap + " items.";

        Map<String,Object> body = new HashMap<>();
        body.put("model", (hintsModel != null && !hintsModel.isBlank()) ? hintsModel : model);
        body.put("prompt", prompt);
        body.put("system", sys);
        body.put("stream", false);
        Map<String,Object> opts = new HashMap<>();
        opts.put("temperature", 0.1);
        opts.put("num_predict", 128);
        body.put("options", opts);

        try {
            JsonNode node = http.post()
                    .uri("/api/generate")
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(body)
                    .retrieve()
                    .body(JsonNode.class);

            String raw = node != null && node.has("response") ? node.get("response").asText() : "";
            String json = cleanToJsonArray(raw);
            if (json == null) return fallbackHeuristic(userText, cap);

            JsonNode arr = mapper.readTree(json);
            if (!arr.isArray()) return fallbackHeuristic(userText, cap);

            LinkedHashSet<String> out = new LinkedHashSet<>();
            for (JsonNode el : arr) {
                if (!el.isTextual()) continue;
                String s = el.asText().trim();
                if (!s.isEmpty()) out.add(s);
                if (out.size() >= cap) break;
            }
            return new ArrayList<>(out);

        } catch (Exception e) {
            return fallbackHeuristic(userText, cap);
        }
    }

    // --- helpers ---

    // Accept raw model output and try to isolate a JSON array: […],
    // stripping code fences etc.
    private static String cleanToJsonArray(String s) {
        if (s == null) return null;
        String t = s.trim();
        // strip fences ```json ... ```
        if (t.startsWith("```")) {
            t = t.replaceAll("^```(?:json)?\\s*|\\s*```$", "");
        }
        // find first [ ... ] block
        Matcher m = Pattern.compile("\\[(?s).*\\]").matcher(t);
        if (m.find()) {
            String block = t.substring(m.start(), m.end());
            return block.trim();
        }
        return null;
    }

    // Lightweight fallback if the model returns junk: capitalized tokens + “like a/an X”
    private static List<String> fallbackHeuristic(String text, int cap) {
        LinkedHashSet<String> out = new LinkedHashSet<>();

        // Proper-noun sequences
        Matcher caps = Pattern.compile("\\b([A-Z][\\p{L}0-9_-]*(?:\\s+[A-Z][\\p{L}0-9_-]*)*)\\b").matcher(text);
        while (caps.find()) {
            String s = caps.group(1).trim();
            if (!STOP.contains(s)) out.add(s);
            if (out.size() >= cap) break;
        }
        // "like a/an X"
        Matcher like = Pattern.compile("\\blike\\s+(?:a|an)\\s+([a-z][a-z-]{2,})\\b", Pattern.CASE_INSENSITIVE).matcher(text);
        while (like.find() && out.size() < cap) {
            String w = like.group(1);
            out.add(capitalize(singularize(w)));
        }
        return new ArrayList<>(out);
    }

    private static String singularize(String w) {
        if (w.endsWith("ies") && w.length() > 3) return w.substring(0, w.length()-3) + "y";
        if (w.endsWith("ses") || w.endsWith("xes")) return w.substring(0, w.length()-2);
        if (w.endsWith("s") && !w.endsWith("ss")) return w.substring(0, w.length()-1);
        return w;
    }
    private static String capitalize(String s) {
        return (s == null || s.isEmpty()) ? s : Character.toUpperCase(s.charAt(0)) + s.substring(1);
    }

    private static final Set<String> STOP = Set.of(
            "Who","What","When","Where","Which","Why","How","Is","Are","Do","Does","Did",
            "And","Or","The","A","An","About","Please","Tell","Me","You","It","This","That",
            "Vs","Than","Smaller","Bigger","Greater","Larger","Less","More","Pokemon","Pokémon"
    );
}
