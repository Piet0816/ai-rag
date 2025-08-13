package com.pseidl.ai.rag.web;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.pseidl.ai.rag.index.InMemoryVectorIndex;
import com.pseidl.ai.rag.index.LibraryIngestionService;
import com.pseidl.ai.rag.index.LibraryIngestionService.Result;
import com.pseidl.ai.rag.library.LibraryFileService;
import com.pseidl.ai.rag.library.LibraryFileService.FileEntry;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.ArraySchema;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;

/**
 * Triggers ingestion (chunk + embed + index) for one file or all files (recursive).
 * Default extensions come from application.properties (app.ingest.extensions).
 */
@RestController
@RequestMapping("/api/ingest")
@Tag(name = "Ingestion", description = "Chunk, embed, and index files from the library folder")
public class IngestionController {

    private static final Logger log = LoggerFactory.getLogger(IngestionController.class);

    private final LibraryIngestionService ingester;
    private final LibraryFileService files;
    private final InMemoryVectorIndex index;

    /** Comma-separated default extensions from configuration. */
    @Value("${app.ingest.extensions:txt,md,csv,json,xml,yaml,yml,properties,java,kt,py,js,ts,tsx,sql,gradle,sh,bat}")
    private String defaultExtCsv;

    public IngestionController(LibraryIngestionService ingester,
                               LibraryFileService files,
                               InMemoryVectorIndex index) {
        this.ingester = ingester;
        this.files = files;
        this.index = index;
    }

    @PostMapping("/file")
    @Operation(
            summary = "Ingest a single file",
            description = "Reads the file (text-like), chunks it, creates embeddings, and updates the in-memory index.",
            responses = {
                    @ApiResponse(responseCode = "200", description = "Ingested",
                            content = @Content(schema = @Schema(implementation = FileIngestResponse.class))),
                    @ApiResponse(responseCode = "400", description = "Invalid request"),
                    @ApiResponse(responseCode = "500", description = "Failed")
            }
    )
    public ResponseEntity<?> ingestOne(
            @Parameter(description = "Relative path under the library root, e.g. people_food.txt")
            @RequestParam("path") String relativePath,
            @RequestParam(name = "chunkSize", defaultValue = "800") int chunkSize,
            @RequestParam(name = "overlap",   defaultValue = "120") int overlap,
            @RequestParam(name = "logEvery",  defaultValue = "50")  int logEvery
    ) {
        try {
            Result r = ingester.ingestTextFile(relativePath, chunkSize, overlap, logEvery);
            return ResponseEntity.ok(new FileIngestResponse(r.source(), r.chunks(), r.totalSeconds(), r.embedSeconds(), r.upsertSeconds(), index.info()));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(new ErrorResponse(safeMsg(e)));
        }
    }

    @PostMapping("/all")
    @Operation(
            summary = "Ingest all files (recursive)",
            description = "Lists files in the library (recursive) and ingests them one by one. "
                        + "Default extensions come from app.ingest.extensions. "
                        + "Use ?ext=txt,md,csv to override.",
            responses = {
                    @ApiResponse(responseCode = "200", description = "Batch ingested",
                            content = @Content(schema = @Schema(implementation = BatchIngestResponse.class))),
                    @ApiResponse(responseCode = "500", description = "Failed")
            }
    )
    public ResponseEntity<?> ingestAll(
            @Parameter(description = "Comma-separated extensions (lowercase, no dot). If empty, uses app.ingest.extensions.")
            @RequestParam(name = "ext", required = false) String extCsv,
            @RequestParam(name = "chunkSize", defaultValue = "800") int chunkSize,
            @RequestParam(name = "overlap",   defaultValue = "120") int overlap,
            @RequestParam(name = "logEvery",  defaultValue = "50")  int logEvery
    ) {
        try {
            Set<String> allowed = parseExtensions(extCsv);
            if (allowed.isEmpty()) {
                allowed = defaultExtensions();
            }

            List<FileEntry> all = files.listFiles(allowed);
            if (all.isEmpty()) {
                return ResponseEntity.ok(new BatchIngestResponse(allowed, 0, 0, 0, List.of(), index.info()));
            }

            int totalFiles = all.size();
            int totalChunks = 0;
            double totalEmbed = 0.0;
            double totalUpsert = 0.0;

            List<FileIngestResponse> perFile = new ArrayList<>(totalFiles);

            int i = 0;
            for (FileEntry fe : all) {
                i++;
                String path = fe.relativePath();
                try {
                    Result r = ingester.ingestTextFile(path, chunkSize, overlap, logEvery);
                    totalChunks += r.chunks();
                    totalEmbed  += r.embedSeconds();
                    totalUpsert += r.upsertSeconds();
                    perFile.add(new FileIngestResponse(r.source(), r.chunks(), r.totalSeconds(), r.embedSeconds(), r.upsertSeconds(), null));
                    log.info("Batch progress {}/{}: '{}', chunks={}, total={}s", i, totalFiles, path, r.chunks(), String.format("%.2f", r.totalSeconds()));
                } catch (Exception e) {
                    log.warn("Failed to ingest '{}': {}", path, e.toString());
                    perFile.add(new FileIngestResponse(path, 0, 0.0, 0.0, 0.0, Map.of("error", safeMsg(e))));
                }
            }

            return ResponseEntity.ok(new BatchIngestResponse(
                    allowed, totalFiles, totalChunks,
                    Math.max(totalEmbed + totalUpsert, 0.0),
                    perFile,
                    index.info()
            ));

        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(new ErrorResponse(safeMsg(e)));
        }
    }

    // --- helpers / DTOs ---

    private Set<String> defaultExtensions() {
        return parseExtensions(defaultExtCsv);
    }

    private static Set<String> parseExtensions(String csv) {
        if (!StringUtils.hasText(csv)) return Collections.emptySet();
        return Arrays.stream(csv.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .map(s -> s.toLowerCase(Locale.ROOT))
                .collect(Collectors.toSet());
    }

    private static String safeMsg(Throwable t) {
        String m = t.getMessage();
        return (m == null || m.isBlank()) ? t.getClass().getSimpleName() : m;
    }

    public record ErrorResponse(String error) {}

    public record FileIngestResponse(
            String source,
            int chunks,
            double totalSeconds,
            double embedSeconds,
            double upsertSeconds,
            Map<String, Object> indexInfo // optional in single-file response; null when returned per-file in batch
    ) {}

    public record BatchIngestResponse(
            Set<String> extensionsUsed,
            int files,
            int chunks,
            double embedPlusUpsertSeconds,
            @ArraySchema(schema = @Schema(implementation = FileIngestResponse.class))
            List<FileIngestResponse> results,
            Map<String, Object> indexInfo
    ) {}
}
