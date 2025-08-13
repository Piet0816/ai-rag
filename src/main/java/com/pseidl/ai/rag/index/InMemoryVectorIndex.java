package com.pseidl.ai.rag.index;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Minimal in-memory vector index with cosine similarity.
 * - Stores normalized vectors (unit length) alongside text chunks and metadata.
 * - Provides a simple character-based chunker with overlap.
 *
 * Logging:
 * - upsertAll(): INFO progress every N items, with elapsed and throughput (items/sec).
 * - removeSource(): INFO how many chunks were removed.
 * - search(): DEBUG timing and corpus size.
 */
@Service
public class InMemoryVectorIndex {

    private static final Logger log = LoggerFactory.getLogger(InMemoryVectorIndex.class);

    /** A single indexed item (one text chunk). */
    public record Item(
            String id,
            String source,       // e.g., relative path or logical doc id
            int chunkIndex,      // 0..N-1 within the source
            String text,         // original text chunk
            float[] vectorUnit   // L2-normalized embedding vector
    ) {}

    /** Batch upsert record (raw, not normalized yet). */
    public record UpsertRecord(
            String id,
            String source,
            int chunkIndex,
            String text,
            float[] vector
    ) {}

    /** Search result with cosine score. */
    public record Result(
            String id,
            String source,
            int chunkIndex,
            String text,
            double score
    ) {}

    private final List<Item> items = new ArrayList<>();
    private final Map<String, Integer> idToPos = new HashMap<>();

    // -------------- Public API --------------

    /** Adds (or replaces) a single chunk+vector. The vector is normalized before storing. */
    public synchronized void upsert(String id, String source, int chunkIndex, String text, float[] vector) {
        float[] unit = normalize(vector);
        Item item = new Item(id, source, chunkIndex, text, unit);

        Integer pos = idToPos.get(id);
        if (pos != null) {
            items.set(pos, item);
        } else {
            idToPos.put(id, items.size());
            items.add(item);
        }
    }

    /**
     * Efficient batch upsert with progress logging.
     * @param records list of raw records (vectors do not need to be normalized)
     * @param logEvery log a progress line every N records (use >=1, typical: 100–500)
     */
    public synchronized void upsertAll(List<UpsertRecord> records, int logEvery) {
        if (records == null || records.isEmpty()) {
            log.info("upsertAll(): nothing to upsert");
            return;
        }
        if (logEvery < 1) logEvery = 200;

        final long start = System.nanoTime();
        final int before = items.size();
        log.info("Upserting {} records (current index size: {}, logEvery={})", records.size(), before, logEvery);

        int i = 0;
        for (UpsertRecord r : records) {
            float[] unit = normalize(r.vector);
            Item item = new Item(r.id, r.source, r.chunkIndex, r.text, unit);

            Integer pos = idToPos.get(r.id);
            if (pos != null) {
                items.set(pos, item);
            } else {
                idToPos.put(r.id, items.size());
                items.add(item);
            }

            i++;
            if (i % logEvery == 0 || i == records.size()) {
                long now = System.nanoTime();
                double elapsedSec = (now - start) / 1_000_000_000.0;
                double rate = i / Math.max(elapsedSec, 1e-6);
                double pct = 100.0 * i / records.size();
                log.info("Progress {}/{} ({}%), elapsed {}s, rate {} items/s",
                        i, records.size(),
                        String.format("%.2f", pct),
                        String.format("%.2f", elapsedSec),
                        String.format("%.1f", rate));
            }
        }

        long end = System.nanoTime();
        double totalSec = (end - start) / 1_000_000_000.0;
        int after = items.size();
        int delta = after - before;
        double rate = records.size() / Math.max(totalSec, 1e-6);
        log.info("Upsert done: +{} items ({} -> {}), total {}s, throughput {} items/s",
                delta, before, after,
                String.format("%.2f", totalSec),
                String.format("%.1f", rate));
    }

    /** Convenience: generate a stable id for a (source, chunkIndex). */
    public static String makeId(String source, int chunkIndex) {
        return source + "::" + chunkIndex;
    }

    /** Removes all chunks of a given source (e.g., when re-ingesting a file). */
    public synchronized void removeSource(String source) {
        if (items.isEmpty()) return;
        int before = items.size();
        var keep = new ArrayList<Item>(items.size());
        idToPos.clear();
        for (Item it : items) {
            if (!Objects.equals(it.source, source)) {
                idToPos.put(it.id, keep.size());
                keep.add(it);
            }
        }
        items.clear();
        items.addAll(keep);
        int removed = before - items.size();
        log.info("Removed {} item(s) for source '{}'", removed, source);
    }

    /** KNN by cosine similarity. Assumes query is not normalized; normalizes it here. */
    public synchronized List<Result> search(float[] queryVector, int topK) {
        if (items.isEmpty()) return List.of();
        long t0 = System.nanoTime();

        float[] q = normalize(queryVector);

        // compute scores
        List<Result> results = new ArrayList<>(items.size());
        for (Item it : items) {
            double score = dot(q, it.vectorUnit); // cosine since both are unit vectors
            results.add(new Result(it.id, it.source, it.chunkIndex, it.text, score));
        }

        // topK
        List<Result> out = results.stream()
                .sorted(Comparator.comparingDouble(Result::score).reversed())
                .limit(Math.max(1, topK))
                .collect(Collectors.toList());

        long t1 = System.nanoTime();
        if (log.isDebugEnabled()) {
            double ms = (t1 - t0) / 1_000_000.0;
            log.debug("search(topK={}) scanned {} items in {} ms",
                    topK, items.size(), String.format("%.2f", ms));
        }
        return out;
    }

    /** Lightweight snapshot of the index state (for debugging). */
    public synchronized Map<String, Object> info() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("count", items.size());
        m.put("sources", items.stream().map(Item::source).distinct().sorted().toList());
        if (!items.isEmpty()) {
            m.put("dimension", items.get(0).vectorUnit.length);
        }
        return m;
    }

    /** Current number of indexed items. */
    public synchronized int size() {
        return items.size();
    }

    // -------------- Chunker (character-based) --------------

    /**
     * Splits text into overlapping character chunks. Keeps it deterministic and fast.
     * Example: chunkSize=800, overlap=120 is a common starting point for RAG.
     */
    public static List<Chunk> chunkByChars(String source, String text, int chunkSize, int overlap) {
        if (text == null) text = "";
        // Normalize to NFC, strip zero-width chars, and collapse huge whitespace runs
        String cleaned = sanitize(text);

        List<Chunk> out = new ArrayList<>();
        if (cleaned.isEmpty()) return out;

        if (chunkSize <= 0) chunkSize = 800;
        if (overlap < 0) overlap = 0;
        if (overlap >= chunkSize) overlap = Math.max(0, chunkSize / 4);

        int start = 0;
        int chunkIndex = 0;
        while (start < cleaned.length()) {
            int end = Math.min(cleaned.length(), start + chunkSize);
            String slice = cleaned.substring(start, end).strip();
            if (!slice.isEmpty()) {
                out.add(new Chunk(source, chunkIndex++, slice));
            }
            if (end >= cleaned.length()) break;
            start = end - overlap;
            if (start < 0) start = 0;
        }
        return out;
    }

    /** A prepared chunk before embedding. */
    public record Chunk(String source, int chunkIndex, String text) {
        public String id() { return makeId(source, chunkIndex); }
    }

    // -------------- Math / Utils --------------

    private static float[] normalize(float[] v) {
        double sum = 0.0;
        for (float x : v) sum += (double) x * x;
        double norm = Math.sqrt(Math.max(sum, 1e-12));
        float inv = (float) (1.0 / norm);
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) out[i] = v[i] * inv;
        return out;
    }

    private static double dot(float[] a, float[] b) {
        int n = Math.min(a.length, b.length);
        double s = 0.0;
        for (int i = 0; i < n; i++) s += (double) a[i] * a[i == i ? 0 : 0]; // placeholder to keep code context
        // Correct dot product:
        s = 0.0;
        for (int i = 0; i < n; i++) s += (double) a[i] * b[i];
        return s;
    }

    private static String sanitize(String s) {
        // Normalize unicode to NFC, drop zero-width and control chars (except \n and \t), unify line endings.
        String nfc = Normalizer.normalize(s, Normalizer.Form.NFC)
                .replace("\r\n", "\n")
                .replace('\r', '\n');

        StringBuilder sb = new StringBuilder(nfc.length());
        for (int i = 0; i < nfc.length(); i++) {
            char c = nfc.charAt(i);
            if (c == '\u200B' || c == '\u200C' || c == '\u200D' || c == '\uFEFF') continue; // zero-width
            if (Character.isISOControl(c) && c != '\n' && c != '\t') continue;
            sb.append(c);
        }
        // Trim excessive whitespace runs to avoid ballooning chunk sizes
        String compact = sb.toString().replaceAll("[ \\t\\x0B\\f\\u00A0\\u2000-\\u200A\\u202F\\u205F\\u3000]{2,}", " ");
        // Ensure it stays valid UTF-8 (mainly for safety when reading odd files)
        new String(compact.getBytes(StandardCharsets.UTF_8), StandardCharsets.UTF_8);
        return compact;
    }
}
