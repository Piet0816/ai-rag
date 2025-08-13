package com.pseidl.ai.rag.web;

import com.pseidl.ai.rag.index.InMemoryVectorIndex;
import com.pseidl.ai.rag.persistence.IndexFileStore;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.ArraySchema;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Map;

/**
 * Read-only index introspection.
 */
@RestController
@RequestMapping("/api/index")
@Tag(name = "Index", description = "In-memory index information and basic store status")
public class IndexInfoController {

    private final InMemoryVectorIndex index;
    private final IndexFileStore store;

    public IndexInfoController(InMemoryVectorIndex index, IndexFileStore store) {
        this.index = index;
        this.store = store;
    }

    @GetMapping("/info")
    @Operation(
            summary = "Get index info",
            description = "Returns the number of chunks, embedding dimension, sources in memory, and basic on-disk store status.",
            responses = { @ApiResponse(responseCode = "200", description = "OK") }
    )
    public ResponseEntity<IndexInfo> info() {
        Map<String, Object> m = index.info();
        int count = ((Number) m.getOrDefault("count", 0)).intValue();
        Integer dimension = (m.containsKey("dimension") ? ((Number) m.get("dimension")).intValue() : null);
        @SuppressWarnings("unchecked")
        List<String> sources = (List<String>) m.getOrDefault("sources", List.of());

        Path storePath = store.getStorePath();
        boolean storeExists = Files.exists(storePath);
        Long size = null;
        Instant lastModified = null;
        try {
            if (storeExists) {
                size = Files.size(storePath);
                lastModified = Files.getLastModifiedTime(storePath).toInstant();
            }
        } catch (Exception ignored) { /* best-effort */ }

        IndexInfo body = new IndexInfo(
                count, dimension, sources,
                storePath.toString(), storeExists, size, lastModified
        );
        return ResponseEntity.ok(body);
    }

    // --- DTO ---

    public record IndexInfo(
            @Schema(description = "Number of indexed chunks") int count,
            @Schema(description = "Embedding vector dimension (null if empty)") Integer dimension,
            @ArraySchema(schema = @Schema(description = "Distinct sources (relative paths)", implementation = String.class))
            List<String> sources,
            @Schema(description = "On-disk store path") String storePath,
            @Schema(description = "Whether the store file exists") boolean storeExists,
            @Schema(description = "Store file size in bytes (if exists)") Long storeSizeBytes,
            @Schema(description = "Store last-modified time (if exists)") Instant storeLastModified
    ) {}
}
