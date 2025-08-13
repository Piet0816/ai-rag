package com.pseidl.ai.rag.persistence;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pseidl.ai.rag.index.InMemoryVectorIndex;
import com.pseidl.ai.rag.index.InMemoryVectorIndex.UpsertRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Simple JSONL(.gz) store for vector index data.
 *
 * Each line is a JSON object:
 * {
 *   "id": "people_food.txt::0",
 *   "source": "people_food.txt",
 *   "chunkIndex": 0,
 *   "text": "Alice likes sushi.",
 *   "vector": [0.12, -0.03, ...]
 * }
 *
 * Usage pattern:
 *  - After ingestion: call rewriteFromBatch(records) to persist the new state (or appendBatch for incremental).
 *  - On startup or on demand: call loadIntoIndex(index, logEvery) to rebuild the in-memory index from disk.
 */
@Component
public class IndexFileStore {

    private static final Logger log = LoggerFactory.getLogger(IndexFileStore.class);

    /** Default file under ./library/.index (overridable in application.properties). */
    @Value("${app.index.store:./library/index.jsonl.gz}")
    private String storePath;

    private final ObjectMapper mapper = new ObjectMapper();

    public Path getStorePath() {
        return Paths.get(storePath).toAbsolutePath().normalize();
    }

    /**
     * Overwrite the store with the given records (atomic via temp + move).
     * Logs progress for large batches.
     */
    public void rewriteFromBatch(List<UpsertRecord> records) throws IOException {
        if (records == null) records = List.of();
        Path target = getStorePath();
        Files.createDirectories(target.getParent());

        Path tmp = target.resolveSibling(target.getFileName() + ".tmp");
        long t0 = System.nanoTime();
        int logEvery = pickLogEvery(records.size());

        log.info("Writing {} records to {}", records.size(), target);
        try (OutputStream fos = Files.newOutputStream(tmp, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
             OutputStream os = isGz(target) ? new GZIPOutputStream(fos, true) : fos;
             BufferedWriter w = new BufferedWriter(new OutputStreamWriter(os, StandardCharsets.UTF_8))) {

            JsonGenerator g = mapper.getFactory().createGenerator(w);
            int i = 0;
            for (UpsertRecord r : records) {
                mapper.writeValue(g, toDto(r));
                w.write('\n');
                i++;
                if (i % logEvery == 0 || i == records.size()) {
                    double sec = (System.nanoTime() - t0) / 1_000_000_000.0;
                    double rate = i / Math.max(sec, 1e-6);
                    log.info("Write progress {}/{} ({}%), elapsed {}s, rate {} rec/s",
                            i, records.size(),
                            String.format("%.1f", 100.0 * i / Math.max(records.size(), 1)),
                            String.format("%.2f", sec),
                            String.format("%.1f", rate));
                }
            }
            g.flush();
        }
        Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);

        double totalSec = (System.nanoTime() - t0) / 1_000_000_000.0;
        log.info("Write complete: {} ({} records) in {}s", target, records.size(), String.format("%.2f", totalSec));
    }

    /**
     * Append to the store (creates file if missing).
     * Good for incremental ingestion when you do not want to rewrite the entire file.
     */
    public void appendBatch(List<UpsertRecord> records) throws IOException {
        if (records == null || records.isEmpty()) return;
        Path target = getStorePath();
        Files.createDirectories(target.getParent());

        long t0 = System.nanoTime();
        int logEvery = pickLogEvery(records.size());

        boolean exists = Files.exists(target);
        log.info("{} {} records to {}", exists ? "Appending" : "Creating", records.size(), target);
        try (OutputStream fos = Files.newOutputStream(target,
                                                      StandardOpenOption.CREATE,
                                                      isGz(target) ? StandardOpenOption.APPEND : StandardOpenOption.APPEND);
             OutputStream os = isGz(target)
                     ? new GZIPOutputStream(new FileOutputStream(target.toFile(), true), true) // append gzip safely
                     : fos;
             BufferedWriter w = new BufferedWriter(new OutputStreamWriter(os, StandardCharsets.UTF_8))) {

            JsonGenerator g = mapper.getFactory().createGenerator(w);
            int i = 0;
            for (UpsertRecord r : records) {
                mapper.writeValue(g, toDto(r));
                w.write('\n');
                i++;
                if (i % logEvery == 0 || i == records.size()) {
                    double sec = (System.nanoTime() - t0) / 1_000_000_000.0;
                    double rate = i / Math.max(sec, 1e-6);
                    log.info("Append progress {}/{} ({}%), elapsed {}s, rate {} rec/s",
                            i, records.size(),
                            String.format("%.1f", 100.0 * i / Math.max(records.size(), 1)),
                            String.format("%.2f", sec),
                            String.format("%.1f", rate));
                }
            }
            g.flush();
        }
        double totalSec = (System.nanoTime() - t0) / 1_000_000_000.0;
        log.info("Append complete: {} ({} records) in {}s", target, records.size(), String.format("%.2f", totalSec));
    }

    /**
     * Stream-load the store into the in-memory index.
     * Uses small batches to bound memory and logs progress.
     */
    public LoadResult loadIntoIndex(InMemoryVectorIndex index, int batchSize, int logEvery) throws IOException {
        Path target = getStorePath();
        if (!Files.exists(target)) {
            log.info("No index store found at {} (nothing to load)", target);
            return new LoadResult(target.toString(), 0, 0.0);
        }

        long t0 = System.nanoTime();
        int total = 0;
        List<UpsertRecord> buffer = new ArrayList<>(Math.max(1, batchSize));

        log.info("Loading index from {}", target);
        try (InputStream fis = Files.newInputStream(target);
             InputStream is = isGz(target) ? new GZIPInputStream(fis) : fis;
             BufferedReader r = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {

            String line;
            while ((line = r.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                StoreDto dto = mapper.readValue(line, StoreDto.class);
                buffer.add(dto.toRecord());

                if (buffer.size() >= batchSize) {
                    index.upsertAll(buffer, logEvery);
                    total += buffer.size();
                    buffer.clear();
                }
            }
            if (!buffer.isEmpty()) {
                index.upsertAll(buffer, logEvery);
                total += buffer.size();
            }
        }
        double sec = (System.nanoTime() - t0) / 1_000_000_000.0;
        log.info("Load complete: {} records loaded into index in {}s", total, String.format("%.2f", sec));
        return new LoadResult(target.toString(), total, sec);
    }

    // --- DTO and helpers ---

    /** JSON shape we persist; separate from UpsertRecord to decouple storage from internal types. */
    private static final class StoreDto {
        public String id;
        public String source;
        public int chunkIndex;
        public String text;
        public float[] vector;

        StoreDto() {} // jackson

        StoreDto(String id, String source, int chunkIndex, String text, float[] vector) {
            this.id = id;
            this.source = source;
            this.chunkIndex = chunkIndex;
            this.text = text;
            this.vector = vector;
        }

        UpsertRecord toRecord() {
            return new UpsertRecord(id, source, chunkIndex, text, vector);
        }
    }

    private static StoreDto toDto(UpsertRecord r) {
        return new StoreDto(r.id(), r.source(), r.chunkIndex(), r.text(), r.vector());
    }

    private static boolean isGz(Path p) {
        String name = p.getFileName().toString().toLowerCase();
        return name.endsWith(".gz");
    }

    private static int pickLogEvery(int size) {
        if (size <= 100) return 25;
        if (size <= 1000) return 100;
        return 250;
    }

    /** Minimal result for load operation. */
    public record LoadResult(String path, int records, double seconds) {}
}
