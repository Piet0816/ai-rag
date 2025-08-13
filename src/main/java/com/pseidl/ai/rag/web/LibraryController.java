package com.pseidl.ai.rag.web;

import com.pseidl.ai.rag.library.LibraryFileService;
import com.pseidl.ai.rag.library.LibraryFileService.FileEntry;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.ArraySchema;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ContentDisposition;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Simple test controller to browse and read files in the library folder.
 */
@RestController
@RequestMapping("/api/library")
@Tag(name = "Library", description = "Browse and read files from the local library folder")
public class LibraryController {

    private static final Logger log = LoggerFactory.getLogger(LibraryController.class);
    private final LibraryFileService files;

    public LibraryController(LibraryFileService files) {
        this.files = files;
    }

    @GetMapping("/files")
    @Operation(
            summary = "List files",
            description = "Lists files under the library root. Optional 'ext' CSV (e.g., txt,md,pdf) filters by extension.",
            responses = {
                    @ApiResponse(responseCode = "200",
                            description = "List of files",
                            content = @Content(array = @ArraySchema(schema = @Schema(implementation = FileEntry.class)))),
                    @ApiResponse(responseCode = "500", description = "Failed to list files")
            }
    )
    public ResponseEntity<?> listFiles(
            @Parameter(description = "Comma-separated list of allowed extensions (lowercase, no dot), e.g. txt,md,pdf")
            @RequestParam(name = "ext", required = false) String extCsv) {
        try {
            Set<String> allowed = parseExtensions(extCsv);
            var list = allowed.isEmpty()
                    ? files.listFiles(Collections.emptySet())      // empty set -> accept all
                    : files.listFiles(allowed);
            return ResponseEntity.ok(list);
        } catch (IOException e) {
            log.warn("Failed to list files", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Failed to list files: " + safeMessage(e));
        }
    }

    @GetMapping("/text")
    @Operation(
            summary = "Read text file",
            description = "Reads a text-like file as UTF-8 plain text by relative path (under the library root).",
            responses = {
                    @ApiResponse(responseCode = "200", description = "Text content",
                            content = @Content(mediaType = "text/plain")),
                    @ApiResponse(responseCode = "400", description = "Bad path"),
                    @ApiResponse(responseCode = "404", description = "Not found")
            }
    )
    public ResponseEntity<?> readText(
            @Parameter(description = "Relative path under the library root") @RequestParam("path") String relativePath) {
        try {
            String content = files.readFile(relativePath);
            return ResponseEntity.ok()
                    .contentType(new MediaType("text", "plain", StandardCharsets.UTF_8))
                    .body(content);
        } catch (IOException e) {
            log.warn("Failed to read text file: {}", relativePath, e);
            return asIoError(e);
        }
    }

    @GetMapping("/bytes")
    @Operation(
            summary = "Read binary file",
            description = "Reads a file as bytes (useful for PDF/DOCX) by relative path.",
            responses = {
                    @ApiResponse(responseCode = "200", description = "Binary content",
                            content = @Content(mediaType = "application/octet-stream")),
                    @ApiResponse(responseCode = "400", description = "Bad path"),
                    @ApiResponse(responseCode = "404", description = "Not found")
            }
    )
    public ResponseEntity<?> readBytes(
            @Parameter(description = "Relative path under the library root") @RequestParam("path") String relativePath) {
        try {
            byte[] data = files.readFileBytes(relativePath);
            String filename = relativePath.contains("/") ? relativePath.substring(relativePath.lastIndexOf('/') + 1) : relativePath;

            HttpHeaders headers = new HttpHeaders();
            headers.setContentDisposition(ContentDisposition.inline().filename(filename).build());
            headers.setContentType(guessContentType(filename));
            headers.setContentLength(data.length);

            return new ResponseEntity<>(data, headers, HttpStatus.OK);
        } catch (IOException e) {
            log.warn("Failed to read binary file: {}", relativePath, e);
            return asIoError(e);
        }
    }

    // --- helpers ---

    private static Set<String> parseExtensions(String csv) {
        if (!StringUtils.hasText(csv)) return Collections.emptySet();
        return Arrays.stream(csv.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .map(s -> s.toLowerCase(Locale.ROOT))
                .collect(Collectors.toSet());
    }

    private static ResponseEntity<String> asIoError(IOException e) {
        String msg = safeMessage(e).toLowerCase(Locale.ROOT);
        HttpStatus status =
                msg.contains("not a regular file") || msg.contains("escapes library root") ? HttpStatus.BAD_REQUEST :
                msg.contains("no such file") || msg.contains("not found") ? HttpStatus.NOT_FOUND :
                HttpStatus.INTERNAL_SERVER_ERROR;
        return ResponseEntity.status(status).body(safeMessage(e));
    }

    private static String safeMessage(Exception e) {
        String m = e.getMessage();
        return m == null ? e.getClass().getSimpleName() : m;
    }

    private static MediaType guessContentType(String filename) {
        String lower = filename.toLowerCase(Locale.ROOT);
        if (lower.endsWith(".pdf")) {
            return MediaType.APPLICATION_PDF;
        }
        if (lower.endsWith(".docx")) {
            return MediaType.parseMediaType("application/vnd.openxmlformats-officedocument.wordprocessingml.document");
        }
        if (lower.endsWith(".doc")) {
            return MediaType.parseMediaType("application/msword");
        }
        if (lower.endsWith(".csv")) {
            return MediaType.parseMediaType("text/csv");
        }
        if (lower.endsWith(".json")) {
            return MediaType.APPLICATION_JSON;
        }
        if (lower.endsWith(".xml")) {
            return MediaType.APPLICATION_XML;
        }
        if (lower.endsWith(".txt") || lower.endsWith(".md") || lower.endsWith(".java") || lower.endsWith(".py")) {
            return new MediaType("text", "plain", StandardCharsets.UTF_8);
        }
        return MediaType.APPLICATION_OCTET_STREAM;
    }
}
