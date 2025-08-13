package com.pseidl.ai.rag.retrieval;

import com.pseidl.ai.rag.index.InMemoryVectorIndex;
import com.pseidl.ai.rag.ollama.EmbeddingClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Multi-query retriever with optional MMR reranking: - Embeds full question +
 * entity hints, unions results, dedupes by id. - If enabled, applies Maximal
 * Marginal Relevance on an overfetched candidate set. - Returns topK hits and a
 * compact context string.
 */
@Service
public class MultiQueryRetrievalService {

	private static final Logger log = LoggerFactory.getLogger(MultiQueryRetrievalService.class);

	// ---- MMR config (overridable via application.properties, but with safe
	// defaults) ----
	@Value("${app.retrieval.mmr.enabled:true}")
	private boolean mmrEnabled;

	@Value("${app.retrieval.mmr.lambda:0.5}") // 1.0 = only relevance, 0.0 = only diversity
	private double mmrLambda;

	@Value("${app.retrieval.mmr.overfetch:24}") // how many candidates to consider before picking topK
	private int mmrOverfetch;

	private final EmbeddingClient embeddings;
	private final InMemoryVectorIndex index;
	private final OllamaEntityHintService hints;

	public MultiQueryRetrievalService(EmbeddingClient embeddings, InMemoryVectorIndex index,
			OllamaEntityHintService hints) { // <
		this.embeddings = embeddings;
		this.index = index;
		this.hints = hints;
	}

	/**
	 * Run multi-query retrieval and build a compact context block.
	 * 
	 * @param userText        natural-language question
	 * @param topK            number of chunks to return after union/dedupe
	 * @param maxContextChars max characters for the context block
	 */
	public RetrievalResult retrieve(String userText, int topK, int maxContextChars) {
		if (userText == null || userText.isBlank()) {
			return new RetrievalResult(List.of(), "(no matches)", List.of());
		}

		// 1) Base query (full sentence)
		float[] qVec = embeddings.embed(userText);
		List<InMemoryVectorIndex.Result> merged = new ArrayList<>(index.search(qVec, Math.max(topK, 6)));

		// 2) Entity hints 
		List<String> hints = safeHintsFromLLM(userText, topK);
		//System.err.println("hints: " + hints);
		int perHint = Math.max(2, topK / Math.max(1, hints.size()));
		for (String h : hints) {
			try {
				float[] hv = embeddings.embed(h);
				merged.addAll(index.search(hv, perHint));
			} catch (Exception e) {
				log.debug("Embedding hint '{}' failed: {}", h, e.toString());
			}
		}

		// 3) Dedupe by id, keep best score
		Map<String, InMemoryVectorIndex.Result> best = new HashMap<>();
		for (var r : merged) {
			var prev = best.get(r.id());
			if (prev == null || r.score() > prev.score())
				best.put(r.id(), r);
		}

		// Candidates sorted by score desc
		List<InMemoryVectorIndex.Result> candidates = new ArrayList<>(best.values());
		candidates.sort(Comparator.comparingDouble(InMemoryVectorIndex.Result::score).reversed());

		// 4) Optional MMR reranking
		List<InMemoryVectorIndex.Result> hits;
		if (mmrEnabled && candidates.size() > topK) {
			int pool = Math.min(Math.max(mmrOverfetch, topK * 3), candidates.size());
			hits = mmrSelect(candidates.subList(0, pool), qVec, topK, mmrLambda);
		} else {
			hits = candidates.size() > topK ? candidates.subList(0, topK) : candidates;
		}

		// 5) Build compact context text
		String context = buildContext(hits, maxContextChars);
		return new RetrievalResult(hits, context, hints);
	}

	// Model-assisted entity extraction (always used; service already falls back
	// internally)
	private List<String> safeHintsFromLLM(String userText, int topK) {
		try {
			int cap = Math.min(8, Math.max(3, topK)); // keep it small
			List<String> list = hints.extractHints(userText, cap);
			return (list != null) ? list : List.of();
		} catch (Exception e) {
			return List.of();
		}
	}

	// --- MMR (Maximal Marginal Relevance) ---
	private List<InMemoryVectorIndex.Result> mmrSelect(List<InMemoryVectorIndex.Result> cand, float[] qVec, int k,
			double lambda) {
		// Pre-embed candidate texts (small set, e.g., <= 24)
		int n = cand.size();
		List<float[]> vecs = new ArrayList<>(n);
		for (var r : cand) {
			try {
				vecs.add(unit(embeddings.embed(r.text())));
			} catch (Exception e) {
				// fallback to zero vector (will rank poorly)
				vecs.add(new float[(qVec != null ? qVec.length : 0)]);
			}
		}
		float[] q = unit(qVec);

		List<InMemoryVectorIndex.Result> selected = new ArrayList<>(k);
		List<float[]> selVecs = new ArrayList<>(k);
		boolean[] used = new boolean[n];

		for (int step = 0; step < k && step < n; step++) {
			double bestScore = -1e9;
			int bestIdx = -1;

			for (int i = 0; i < n; i++) {
				if (used[i])
					continue;
				double rel = cosine(q, vecs.get(i));
				double div = 0.0;
				for (float[] s : selVecs) {
					div = Math.max(div, cosine(vecs.get(i), s));
				}
				double mmr = lambda * rel - (1.0 - lambda) * div;
				if (mmr > bestScore) {
					bestScore = mmr;
					bestIdx = i;
				}
			}

			if (bestIdx < 0)
				break;
			used[bestIdx] = true;
			selected.add(cand.get(bestIdx));
			selVecs.add(vecs.get(bestIdx));
		}
		return selected;
	}

	private static float[] unit(float[] v) {
		if (v == null || v.length == 0)
			return new float[0];
		double s = 0.0;
		for (float x : v)
			s += (double) x * x;
		double n = Math.sqrt(Math.max(s, 1e-12));
		float inv = (float) (1.0 / n);
		float[] out = new float[v.length];
		for (int i = 0; i < v.length; i++)
			out[i] = v[i] * inv;
		return out;
	}

	private static double cosine(float[] a, float[] b) {
		int n = Math.min(a.length, b.length);
		double s = 0.0;
		for (int i = 0; i < n; i++)
			s += (double) a[i] * b[i];
		return s;
	}

	/** Compact, source-labeled context respecting a character budget. */
	private static String buildContext(List<InMemoryVectorIndex.Result> hits, int maxChars) {
		if (hits == null || hits.isEmpty())
			return "(no matches)";
		StringBuilder sb = new StringBuilder(Math.min(maxChars, 8192));
		for (var h : hits) {
			String header = "\n[" + h.source() + "#" + h.chunkIndex() + "] (score "
					+ String.format(Locale.ROOT, "%.3f", h.score()) + ")\n";
			if (sb.length() + header.length() > maxChars)
				break;
			sb.append(header);
			String txt = h.text();
			int remain = maxChars - sb.length();
			if (txt.length() <= remain) {
				sb.append(txt);
			} else {
				sb.append(txt, 0, Math.max(0, remain));
				break;
			}
		}
		return sb.toString();
	}

	// --- DTO ---
	public record RetrievalResult(List<InMemoryVectorIndex.Result> hits, String context, List<String> hintsUsed) {
		public List<RetrievedHit> toRetrieved() {
			return hits == null ? List.of()
					: hits.stream().map(h -> new RetrievedHit(h.source(), h.chunkIndex(), h.score()))
							.collect(Collectors.toList());
		}
	}

	/** Lightweight public view for UI/debug. */
	public record RetrievedHit(String source, int chunkIndex, double score) {
	}
}
