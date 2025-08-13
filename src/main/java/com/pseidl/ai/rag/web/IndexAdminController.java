package com.pseidl.ai.rag.web;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Map;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.pseidl.ai.rag.index.InMemoryVectorIndex;
import com.pseidl.ai.rag.index.LibraryWatchScheduler;
import com.pseidl.ai.rag.persistence.IndexFileStore;
import com.pseidl.ai.rag.persistence.IndexFileStore.LoadResult;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;

/**
 * Admin endpoints for persisting and restoring the index.
 */
@RestController
@RequestMapping("/api/index")
@Tag(name = "Index Admin", description = "Persist and restore the in-memory index")
public class IndexAdminController {

    private final InMemoryVectorIndex index;
    private final IndexFileStore store;
    private final LibraryWatchScheduler scheduler;

    public IndexAdminController(InMemoryVectorIndex index,
                                IndexFileStore store,
                                LibraryWatchScheduler scheduler) {
        this.index = index;
        this.store = store;
        this.scheduler = scheduler;
    }

    @PostMapping("/save")
    @Operation(
            summary = "Save now (compact store)",
            description = "Runs the same compaction as the nightly job: deduplicates by chunk id and drops entries for deleted files."
    )
    public ResponseEntity<SaveNowResponse> saveNow() {
        Path path = store.getStorePath();

        long beforeSize = safeSize(path);
        Instant beforeTime = safeMtime(path);

        // Reuse the scheduler's compaction logic synchronously.
        scheduler.compactStore();

        long afterSize = safeSize(path);
        Instant afterTime = safeMtime(path);

        // Snapshot current in-memory state for convenience
        Map<String, Object> info = index.info();
        int count = ((Number) info.getOrDefault("count", 0)).intValue();
        @SuppressWarnings("unchecked")
        List<String> sources = (List<String>) info.getOrDefault("sources", List.of());
        Integer dim = info.containsKey("dimension") ? ((Number) info.get("dimension")).intValue() : null;

        SaveNowResponse body = new SaveNowResponse(
                path.toString(),
                beforeSize, afterSize,
                beforeTime, afterTime,
                count, dim, sources
        );
        return ResponseEntity.ok(body);
    }

    @PostMapping("/load")
    @Operation(
            summary = "Load now (rebuild in-memory index from disk)",
            description = "Loads the persisted store into memory. Optionally clears the in-memory index first."
    )
    public ResponseEntity<LoadNowResponse> loadNow(
            @RequestParam(name = "batchSize", defaultValue = "200") int batchSize,
            @RequestParam(name = "logEvery",  defaultValue = "100") int logEvery,
            @RequestParam(name = "clear",     defaultValue = "true") boolean clear
    ) {
        if (clear) {
            // Remove all current sources before loading
            @SuppressWarnings("unchecked")
            List<String> sources = (List<String>) index.info().getOrDefault("sources", List.of());
            for (String s : sources) {
                index.removeSource(s);
            }
        }

        try {
            LoadResult r = store.loadIntoIndex(index, Math.max(1, batchSize), Math.max(1, logEvery));
            Map<String, Object> info = index.info();
            int count = ((Number) info.getOrDefault("count", 0)).intValue();
            Integer dim = info.containsKey("dimension") ? ((Number) info.get("dimension")).intValue() : null;

            LoadNowResponse body = new LoadNowResponse(
                    r.path(), r.records(), r.seconds(),
                    count, dim
            );
            return ResponseEntity.ok(body);
        } catch (Exception e) {
            return ResponseEntity.internalServerError()
                    .body(new LoadNowResponse(store.getStorePath().toString(), 0, 0.0,
                            ((Number) index.info().getOrDefault("count", 0)).intValue(),
                            (Integer) index.info().get("dimension")));
        }
    }

    // --- helpers / DTOs ---

    private static long safeSize(Path p) {
        try { return (p != null && Files.exists(p)) ? Files.size(p) : -1L; }
        catch (Exception ignored) { return -1L; }
    }

    private static Instant safeMtime(Path p) {
        try { return (p != null && Files.exists(p)) ? Files.getLastModifiedTime(p).toInstant() : null; }
        catch (Exception ignored) { return null; }
    }

    public record SaveNowResponse(
            String storePath,
            long sizeBytesBefore,
            long sizeBytesAfter,
            Instant lastModifiedBefore,
            Instant lastModifiedAfter,
            int inMemoryChunkCount,
            Integer embeddingDimension,
            List<String> inMemorySources
    ) {}

    public record LoadNowResponse(
            String storePath,
            int loadedRecords,
            double seconds,
            int inMemoryChunkCount,
            Integer embeddingDimension
    ) {}
}
