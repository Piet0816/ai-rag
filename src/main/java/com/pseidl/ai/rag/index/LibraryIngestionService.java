package com.pseidl.ai.rag.index;

import static com.pseidl.ai.rag.index.InMemoryVectorIndex.chunkByChars;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import com.pseidl.ai.rag.config.AppConfig;
import com.pseidl.ai.rag.ollama.EmbeddingClient;
import com.pseidl.ai.rag.persistence.IndexFileStore;

/**
 * Ingests a single library file:
 *  - extract text (PDF/DOC/DOCX via Tika, plain files via UTF-8),
 *  - chunk (character-based with overlap),
 *  - embed,
 *  - upsert to in-memory index and append to on-disk store.
 *
 * Returns timing breakdown that your controller expects.
 */
@Service
public class LibraryIngestionService {

    private static final Logger log = LoggerFactory.getLogger(LibraryIngestionService.class);

    private final AppConfig app;
    private final TextExtractionService extractor;
    private final EmbeddingClient embeddings;
    private final InMemoryVectorIndex index;
    private final IndexFileStore store;

    public LibraryIngestionService(AppConfig app,
                                   TextExtractionService extractor,
                                   EmbeddingClient embeddings,
                                   InMemoryVectorIndex index,
                                   IndexFileStore store) {
        this.app = app;
        this.extractor = extractor;
        this.embeddings = embeddings;
        this.index = index;
        this.store = store;
    }

    /**
     * Ingest a single file by its relative path inside the library folder.
     */
    public Result ingestTextFile(String relativePath, int chunkSize, int overlap, int logEvery) throws Exception {
        final long t0 = System.nanoTime();

        Path root = Paths.get(app.getLibraryDir()).toAbsolutePath().normalize();
        Path abs = root.resolve(relativePath).normalize();

        if (!abs.startsWith(root)) {
            throw new IllegalArgumentException("Refusing to read outside library dir: " + relativePath);
        }
        if (!Files.exists(abs)) {
            throw new IllegalArgumentException("File not found: " + relativePath);
        }

        final String source = root.relativize(abs).toString().replace('\\', '/');

        // 1) Extract text
        String text = extractor.extract(abs);
        if (text == null || text.isBlank()) {
            log.info("No textual content extracted from '{}'; skipping.", source);
            return new Result(source, 0, secondsSince(t0), 0.0, 0.0);
        }

        // 2) Chunk
        List<InMemoryVectorIndex.Chunk> chunks = chunkByChars(source, text, chunkSize, overlap);
        if (chunks.isEmpty()) {
            return new Result(source, 0, secondsSince(t0), 0.0, 0.0);
        }

        // 3) Embed
        long tEmbed0 = System.nanoTime();
        List<InMemoryVectorIndex.UpsertRecord> batch = new ArrayList<>(chunks.size());
        int i = 0;
        int every = Math.max(1, logEvery);
        for (var ch : chunks) {
            float[] vec = embeddings.embed(ch.text());
            String id = InMemoryVectorIndex.makeId(source, ch.chunkIndex());
            batch.add(new InMemoryVectorIndex.UpsertRecord(id, source, ch.chunkIndex(), ch.text(), vec));
            i++;
            if (i % every == 0) {
                double sec = secondsSince(tEmbed0);
                log.info("Embedding '{}'… {}/{} chunks ({}s)", source, i, chunks.size(), fmt2(sec));
            }
        }
        double embedSeconds = secondsSince(tEmbed0);

        // 4) Upsert RAM + append to disk
        long tUpsert0 = System.nanoTime();
        index.upsertAll(batch, every);
        store.appendBatch(batch);
        double upsertSeconds = secondsSince(tUpsert0);

        double totalSeconds = secondsSince(t0);
        log.info("Ingested '{}' (chunks={}) in {}s (embed={}s, upsert={}s)",
                source, chunks.size(), fmt2(totalSeconds), fmt2(embedSeconds), fmt2(upsertSeconds));

        return new Result(source, chunks.size(), totalSeconds, embedSeconds, upsertSeconds);
    }

    // ---------- util ----------

    private static double secondsSince(long t0) {
        return Duration.ofNanos(System.nanoTime() - t0).toNanos() / 1_000_000_000.0;
    }

    private static String fmt2(double v) { return String.format(Locale.ROOT, "%.2f", v); }

    // ---------- DTO your controller expects ----------
    public record Result(
            String source,
            int chunks,
            double totalSeconds,
            double embedSeconds,
            double upsertSeconds
    ) {}
}
