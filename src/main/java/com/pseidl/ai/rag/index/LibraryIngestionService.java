package com.pseidl.ai.rag.index;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import com.pseidl.ai.rag.config.AppConfig;
import com.pseidl.ai.rag.index.InMemoryVectorIndex.Chunk;
import com.pseidl.ai.rag.index.InMemoryVectorIndex.UpsertRecord;
import com.pseidl.ai.rag.library.LibraryFileService;
import com.pseidl.ai.rag.ollama.EmbeddingClient;
import com.pseidl.ai.rag.persistence.IndexFileStore;

/**
 * Ingests a single text-like file from the library:
 *  1) Read text
 *  2) Chunk by characters (with overlap)
 *  3) Create embeddings (logs progress)
 *  4) Upsert into the in-memory vector index (index logs progress + total time)
 *  5) Persist the new chunks to disk via IndexFileStore (append mode by default)
 *
 * NOTE: For PDF/DOCX we will later add text extraction. For now use text/CSV/MD/etc.
 */
@Service
public class LibraryIngestionService {

    private static final Logger log = LoggerFactory.getLogger(LibraryIngestionService.class);

    private final LibraryFileService files;
    private final EmbeddingClient embeddings;
    private final InMemoryVectorIndex index;
    private final IndexFileStore store;

    /**
     * Toggle automatic persistence after a successful ingest.
     * Default true if not set: app.index.auto-save=true|false
     */
    @Value("${app.index.auto-save:true}")
    private boolean autoSave;

    public LibraryIngestionService(LibraryFileService files,
                                   EmbeddingClient embeddings,
                                   InMemoryVectorIndex index,                              
                                   IndexFileStore store) {
        this.files = files;
        this.embeddings = embeddings;
        this.index = index;
        this.store = store;
    }

    /**
     * Ingests one file (relative to the library root).
     * Typical starting values: chunkSize=800, overlap=120, logEvery=50.
     */
    public Result ingestTextFile(String relativePath, int chunkSize, int overlap, int logEvery) throws Exception {
        long t0 = System.nanoTime();
        String source = relativePath.replace('\\', '/');

        // 1) Read
        String text = files.readFile(relativePath);
        if (text == null || text.isBlank()) {
            index.removeSource(source);
            // We intentionally do not compact the on-disk store here (append-only).
            // Next step: implement "upsert-by-source" to rewrite the store without this source.
            return new Result(source, 0, 0, 0, 0);
        }

        // 2) Chunk
        List<Chunk> chunks = InMemoryVectorIndex.chunkByChars(source, text, chunkSize, overlap);
        log.info("Ingesting '{}' -> {} chunk(s) [chunkSize={}, overlap={}]", source, chunks.size(), chunkSize, overlap);
        if (chunks.isEmpty()) {
            index.removeSource(source);
            long tEmpty = System.nanoTime();
            return new Result(source, 0, secs(tEmpty - t0), 0, 0);
        }

        // 3) Embed (with progress logging)
        if (logEvery < 1) logEvery = 50;
        long te0 = System.nanoTime();
        List<UpsertRecord> batch = new ArrayList<>(chunks.size());
        int i = 0;
        for (Chunk c : chunks) {
            float[] vec = embeddings.embed(c.text());
            batch.add(new UpsertRecord(c.id(), c.source(), c.chunkIndex(), c.text(), vec));

            i++;
            if (i % logEvery == 0 || i == chunks.size()) {
                long now = System.nanoTime();
                double elapsed = secs(now - te0);
                double rate = i / Math.max(elapsed, 1e-6);
                double pct = 100.0 * i / chunks.size();
                log.info("Embedding progress {}/{} ({}%), elapsed {}s, rate {} items/s",
                        i, chunks.size(),
                        String.format("%.1f", pct),
                        String.format("%.2f", elapsed),
                        String.format("%.1f", rate));
            }
        }
        long te1 = System.nanoTime();

        // 4) Replace old entries in-memory for this source, then upsert new ones
        index.removeSource(source);
        long tu0 = System.nanoTime();
        index.upsertAll(batch, Math.max(1, logEvery));
        long tu1 = System.nanoTime();

        // 5) Persist to disk (append mode)
        // NOTE: This will append even if the same source was ingested before.
        // In the next step we can add a "compact/upsert-by-source" operation to avoid duplicates on disk.
        if (autoSave) {
            try {
                long ts0 = System.nanoTime();
                store.appendBatch(batch);
                long ts1 = System.nanoTime();
                log.info("Persisted {} record(s) for '{}' to {}", batch.size(), source, store.getStorePath());
                log.info("Persistence time: {}s", String.format("%.2f", secs(ts1 - ts0)));
            } catch (Exception e) {
                log.warn("Failed to persist records for '{}': {}", source, e.toString());
            }
        } else {
            log.info("Auto-save is disabled (app.index.auto-save=false); skipping persistence for '{}'", source);
        }

        long t1 = System.nanoTime();
        Result r = new Result(
                source,
                chunks.size(),
                secs(t1 - t0),
                secs(te1 - te0),
                secs(tu1 - tu0)
        );
        log.info("Ingestion complete for '{}': chunks={}, total={}s (embed={}s, upsert={}s)",
                r.source(), r.chunks(), fmt(r.totalSeconds()), fmt(r.embedSeconds()), fmt(r.upsertSeconds()));
        return r;
    }

    private static double secs(long nanos) {
        return nanos / 1_000_000_000.0;
    }

    private static String fmt(double v) {
        return String.format("%.2f", v);
    }

    /** Minimal ingest result for metrics. */
    public record Result(
            String source,
            int chunks,
            double totalSeconds,
            double embedSeconds,
            double upsertSeconds
    ) {
        public String source() { return source; }
        public int chunks() { return chunks; }
        public String totalPretty() { return Duration.ofMillis((long)(totalSeconds * 1000)).toString(); }
    }
}
