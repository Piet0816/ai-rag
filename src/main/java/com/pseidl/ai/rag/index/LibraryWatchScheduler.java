package com.pseidl.ai.rag.index;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.zip.GZIPInputStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.scheduling.annotation.Scheduled;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.pseidl.ai.rag.index.InMemoryVectorIndex.UpsertRecord;
import com.pseidl.ai.rag.library.LibraryFileService;
import com.pseidl.ai.rag.library.LibraryFileService.FileEntry;
import com.pseidl.ai.rag.persistence.IndexFileStore;

/**
 * Periodically scans the library folder and:
 *  - Ingests new or changed files (by mtime/size).
 *  - Removes index entries for deleted files.
 *  - Runs a nightly compaction of the on-disk store to deduplicate and drop removed sources.
 *
 * Compaction now delegates writing to IndexFileStore.rewriteFromBatch() to avoid GZIP stream issues.
 */
@Configuration
@EnableScheduling
public class LibraryWatchScheduler {

    private static final Logger log = LoggerFactory.getLogger(LibraryWatchScheduler.class);

    private final LibraryFileService files;
    private final LibraryIngestionService ingester;
    private final InMemoryVectorIndex index;
    private final IndexFileStore store;
    private final ObjectMapper mapper = new ObjectMapper();

    // --- Configurable via application.properties ---

    @Value("${app.ingest.extensions:txt,md,csv,json,xml,yaml,yml,properties,java,kt,py,js,ts,tsx,sql,gradle,sh,bat}")
    private String extCsv;

    @Value("${app.ingest.chunk-size:800}")
    private int chunkSize;

    @Value("${app.ingest.overlap:120}")
    private int overlap;

    @Value("${app.ingest.log-every:50}")
    private int logEvery;

    @Value("${app.ingest.watch.enabled:true}")
    private boolean watchEnabled;

    @Value("${app.ingest.watch.delay-ms:60000}")
    private long watchDelayMs;

    @Value("${app.index.compact.enabled:true}")
    private boolean compactEnabled;

    @Value("${app.index.compact.cron:0 0 3 * * *}")
    private String compactCron; // info only

    // --- Internal state for change detection ---
    private final Map<String, FileMeta> state = new ConcurrentHashMap<>();
    private volatile boolean scanRunning = false;
    private volatile boolean compactRunning = false;

    public LibraryWatchScheduler(LibraryFileService files,
                                 LibraryIngestionService ingester,
                                 InMemoryVectorIndex index,
                                 IndexFileStore store) {
        this.files = files;
        this.ingester = ingester;
        this.index = index;
        this.store = store;
    }

    // -------- Periodic scan (new/changed/deleted) --------

    @Scheduled(fixedDelayString = "${app.ingest.watch.delay-ms:60000}")
    public void scanAndIngest() {
        if (!watchEnabled) return;
        if (scanRunning) return;
        scanRunning = true;
        long t0 = System.nanoTime();

        try {
            Set<String> allowed = parseExtensions(extCsv);
            List<FileEntry> current = files.listFiles(allowed); // recursive
            Set<String> seen = new HashSet<>(current.size());

            int newOrChanged = 0;
            for (FileEntry fe : current) {
                String path = fe.relativePath();
                seen.add(path);
                FileMeta meta = new FileMeta(fe.size(), fe.lastModified());
                FileMeta prev = state.get(path);

                if (prev == null || !prev.equals(meta)) {
                    log.info("Detected {} file: '{}'", (prev == null ? "new" : "changed"), path);
                    try {
                        var res = ingester.ingestTextFile(path, chunkSize, overlap, logEvery);
                        state.put(path, meta);
                        newOrChanged++;
                        log.info("Ingested '{}': {} chunks in {}s", path, res.chunks(), fmt(res.totalSeconds()));
                    } catch (Exception e) {
                        log.warn("Failed to ingest '{}': {}", path, e.toString());
                    }
                }
            }

            // Deletions
            int removed = 0;
            for (String known : new ArrayList<>(state.keySet())) {
                if (!seen.contains(known)) {
                    index.removeSource(known);
                    state.remove(known);
                    removed++;
                    log.info("Removed index for deleted file '{}'", known);
                }
            }

            double sec = (System.nanoTime() - t0) / 1_000_000_000.0;
            if (newOrChanged > 0 || removed > 0) {
                log.info("Watch scan done: {} updated/new, {} removed, in {}s", newOrChanged, removed, fmt(sec));
            } else {
                log.debug("Watch scan: no changes ({}s)", fmt(sec));
            }

        } catch (Exception e) {
            log.warn("Watch scan failed: {}", e.toString());
        } finally {
            scanRunning = false;
        }
    }

    // -------- Nightly compaction (dedupe + drop missing sources) --------

    @Scheduled(cron = "${app.index.compact.cron:0 0 3 * * *}")
    public void compactStore() {
        if (!compactEnabled) return;
        if (compactRunning) return;
        compactRunning = true;

        final Path path = store.getStorePath();
        long t0 = System.nanoTime();

        try {
            if (!Files.exists(path)) {
                log.info("Compaction: no store at {} (skipped)", path);
                return;
            }

            // Build set of live sources from current library
            Set<String> allowed = parseExtensions(extCsv);
            Set<String> liveSources = new HashSet<>();
            for (FileEntry fe : files.listFiles(allowed)) {
                liveSources.add(fe.relativePath());
            }

            // Read all lines, keep only the LAST occurrence per id, only if source still exists.
            int inLines = 0;
            Map<String, UpsertRecord> latest = new LinkedHashMap<>();
            try (InputStream fis = Files.newInputStream(path);
                 InputStream is = isGz(path) ? new GZIPInputStream(fis) : fis;
                 BufferedReader br = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {

                String line;
                while ((line = br.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty()) continue;
                    inLines++;

                    StoreLine sl;
                    try {
                        sl = mapper.readValue(line, StoreLine.class);
                    } catch (Exception ex) {
                        log.warn("Compaction: malformed line skipped: {}", ex.toString());
                        continue;
                    }
                    if (sl.id == null || sl.source == null) continue;
                    if (!liveSources.contains(sl.source)) continue;

                    // last write wins
                    latest.put(sl.id, new UpsertRecord(sl.id, sl.source, sl.chunkIndex, sl.text, sl.vector));
                }
            }

            // Rewrite compacted store atomically via IndexFileStore (prevents GZIP stream issues)
            List<UpsertRecord> compacted = new ArrayList<>(latest.values());
            store.rewriteFromBatch(compacted);

            double sec = (System.nanoTime() - t0) / 1_000_000_000.0;
            log.info("Compaction complete: {} -> {} lines ({} sources), in {}s",
                    inLines, compacted.size(), distinctSources(compacted), fmt(sec));

        } catch (Exception e) {
            log.warn("Compaction failed: {}", e.toString());
        } finally {
            compactRunning = false;
        }
    }

    // -------- Helpers --------

    private static Set<String> parseExtensions(String csv) {
        if (csv == null || csv.isBlank()) return Set.of();
        Set<String> out = new LinkedHashSet<>();
        for (String s : csv.split(",")) {
            String t = s.trim().toLowerCase(Locale.ROOT);
            if (!t.isEmpty()) out.add(t);
        }
        return out;
    }

    private static boolean isGz(Path p) {
        String name = p.getFileName().toString().toLowerCase(Locale.ROOT);
        return name.endsWith(".gz");
    }

    private static String fmt(double v) {
        return String.format("%.2f", v);
    }

    private static int distinctSources(Collection<UpsertRecord> recs) {
        Set<String> s = new HashSet<>();
        for (UpsertRecord r : recs) s.add(r.source());
        return s.size();
    }

    /** Metadata we track for change detection. */
    private record FileMeta(long size, Instant lastModified) {}

    /** Minimal shape of a persisted line in the JSONL(.gz) store. */
    private static final class StoreLine {
        public String id;
        public String source;
        public int chunkIndex;
        public String text;
        public float[] vector;
        public StoreLine() {}
    }
}
