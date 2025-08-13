package com.pseidl.ai.rag.retrieval;

import com.pseidl.ai.rag.index.InMemoryVectorIndex;
import com.pseidl.ai.rag.ollama.EmbeddingClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Multi-query retriever:
 *  - Embeds the full user question AND detected entity tokens (proper nouns),
 *  - Retrieves for all queries, unions results, keeps best score per chunk,
 *  - Sorts by score and returns topK, plus a compact text context.
 */
@Service
public class MultiQueryRetrievalService {

    private static final Logger log = LoggerFactory.getLogger(MultiQueryRetrievalService.class);

    private static final Pattern PROPER_NOUN = Pattern.compile("\\b([A-Z][A-Za-z0-9_-]{2,})\\b");
    private static final Set<String> STOP = Set.of(
            "Who","What","When","Where","Which","Why","How","Is","Are","Do","Does","Did",
            "And","Or","The","A","An","About","Please","Tell","Me","You","It","This","That",
            "Vs","vs","Than","Smaller","Bigger","Greater","Larger","Less","More"
    );

    private final EmbeddingClient embeddings;
    private final InMemoryVectorIndex index;

    public MultiQueryRetrievalService(EmbeddingClient embeddings, InMemoryVectorIndex index) {
        this.embeddings = embeddings;
        this.index = index;
    }

    /**
     * Run multi-query retrieval and build a compact context block.
     * @param userText natural-language question
     * @param topK number of chunks to return after union/dedupe
     * @param maxContextChars max characters for the context block
     */
    public RetrievalResult retrieve(String userText, int topK, int maxContextChars) {
        if (userText == null || userText.isBlank()) {
            return new RetrievalResult(List.of(), "(no matches)", List.of());
        }

        // 1) Base query (full sentence)
        float[] qVec = embeddings.embed(userText);
        List<InMemoryVectorIndex.Result> merged = new ArrayList<>(index.search(qVec, Math.max(topK, 6)));

        // 2) Entity hints (Pikachu, Charmander, etc.)
        List<String> hints = extractEntityHints(userText);
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
            if (prev == null || r.score() > prev.score()) best.put(r.id(), r);
        }

        // 4) Sort + cap
        List<InMemoryVectorIndex.Result> hits = new ArrayList<>(best.values());
        hits.sort(Comparator.comparingDouble(InMemoryVectorIndex.Result::score).reversed());
        if (hits.size() > topK) hits = hits.subList(0, topK);

        // 5) Build compact context text
        String context = buildContext(hits, maxContextChars);

        return new RetrievalResult(hits, context, hints);
    }

    // --- helpers ---

    private static List<String> extractEntityHints(String text) {
        var out = new LinkedHashSet<String>();
        var m = PROPER_NOUN.matcher(text);
        while (m.find()) {
            String tok = m.group(1);
            if (!STOP.contains(tok)) out.add(tok);
        }
        return new ArrayList<>(out);
    }

    /** Compact, source-labeled context respecting a character budget. */
    private static String buildContext(List<InMemoryVectorIndex.Result> hits, int maxChars) {
        if (hits == null || hits.isEmpty()) return "(no matches)";
        StringBuilder sb = new StringBuilder(Math.min(maxChars, 8192));
        for (var h : hits) {
            String header = "\n[" + h.source() + "#" + h.chunkIndex() + "] (score " +
                    String.format(Locale.ROOT, "%.3f", h.score()) + ")\n";
            if (sb.length() + header.length() > maxChars) break;
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
    public record RetrievalResult(
            List<InMemoryVectorIndex.Result> hits,
            String context,
            List<String> hintsUsed
    ) {
        public List<RetrievedHit> toRetrieved() {
            return hits == null ? List.of() :
                    hits.stream()
                        .map(h -> new RetrievedHit(h.source(), h.chunkIndex(), h.score()))
                        .collect(Collectors.toList());
        }
    }

    /** Lightweight public view for UI/debug. */
    public record RetrievedHit(String source, int chunkIndex, double score) {}
}
