package com.pseidl.ai.rag.web;

import com.pseidl.ai.rag.config.AppConfig;
import com.pseidl.ai.rag.ollama.EmbeddingClient;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

/**
 * Minimal endpoint to test embedding generation.
 * POST a short text and get back vector dimension + a small preview.
 */
@RestController
@RequestMapping("/api/embedding")
@Tag(name = "Embedding", description = "Test endpoint for Ollama embeddings")
public class EmbeddingController {

    private final EmbeddingClient embeddings;
    private final AppConfig app;

    public EmbeddingController(EmbeddingClient embeddings, AppConfig app) {
        this.embeddings = embeddings;
        this.app = app;
    }

    @PostMapping(value = "/test", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    @Operation(
            summary = "Create an embedding for a short text",
            description = "Calls the local Ollama embedding model and returns dimension, preview values, and L2 norm.",
            responses = {
                    @ApiResponse(responseCode = "200", description = "Embedding created",
                            content = @Content(schema = @Schema(implementation = EmbedResponse.class))),
                    @ApiResponse(responseCode = "400", description = "Invalid request",
                            content = @Content(schema = @Schema(implementation = ErrorResponse.class))),
                    @ApiResponse(responseCode = "500", description = "Embedding failed",
                            content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
            }
    )
    public ResponseEntity<?> embed(@RequestBody EmbedRequest req) {
        if (req == null || req.text() == null || req.text().isBlank()) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body(new ErrorResponse("Missing 'text'"));
        }
        try {
            float[] vec = embeddings.embed(req.text());
            int dim = vec.length;
            float[] preview = preview(vec, 12);
            double norm = l2(vec);
            String model = app.getOllama().getEmbeddingModel();
            return ResponseEntity.ok(new EmbedResponse(model, dim, preview, norm));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(new ErrorResponse(e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage()));
        }
    }

    // --- helpers ---

    private static float[] preview(float[] v, int n) {
        int k = Math.min(n, v.length);
        float[] out = new float[k];
        System.arraycopy(v, 0, out, 0, k);
        return out;
    }

    private static double l2(float[] v) {
        double sum = 0.0;
        for (float x : v) sum += (double) x * x;
        return Math.sqrt(sum);
    }

    // --- DTOs ---

    public record EmbedRequest(
            @Schema(description = "Text to embed", example = "Alice likes sushi.")
            String text
    ) {}

    public record EmbedResponse(
            @Schema(description = "Embedding model used") String model,
            @Schema(description = "Vector dimension") int dimension,
            @Schema(description = "First few values of the vector") float[] preview,
            @Schema(description = "L2 norm of the vector") double normL2
    ) {}

    public record ErrorResponse(
            @Schema(description = "Error message") String error
    ) {}
}
