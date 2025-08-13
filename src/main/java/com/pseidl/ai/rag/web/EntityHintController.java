package com.pseidl.ai.rag.web;

import com.pseidl.ai.rag.retrieval.OllamaEntityHintService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.ArraySchema;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/hints")
@Tag(name = "Hints", description = "Extract entity hints from a prompt using the local Ollama model")
public class EntityHintController {

    private final OllamaEntityHintService hints;

    public EntityHintController(OllamaEntityHintService hints) {
        this.hints = hints;
    }

    @PostMapping(
            value = "/extract",
            consumes = MediaType.APPLICATION_JSON_VALUE,
            produces = MediaType.APPLICATION_JSON_VALUE
    )
    @Operation(
            summary = "Extract entity hints",
            description = "Asks the local Ollama model to return a JSON array of short entity hints for the given text. " +
                          "Useful for multi-query retrieval (e.g., returns [\"Elephant\",\"Rhino\",\"Africe\"]).",
            responses = {
                    @ApiResponse(responseCode = "200", description = "OK"),
                    @ApiResponse(responseCode = "400", description = "Missing or empty text")
            }
    )
    public ResponseEntity<HintResponse> extract(
            @RequestBody HintRequest body,
            @RequestParam(name = "max", required = false, defaultValue = "6") int max
    ) {
        if (body == null || !StringUtils.hasText(body.text())) {
            return ResponseEntity.badRequest().body(new HintResponse(List.of(), 0, "(empty input)"));
        }
        long t0 = System.nanoTime();
        List<String> out = hints.extractHints(body.text(), Math.max(1, max));
        long ms = (System.nanoTime() - t0) / 1_000_000;
        return ResponseEntity.ok(new HintResponse(out, ms, body.text()));
    }

    // --- DTOs ---

    public record HintRequest(
            @Schema(description = "User prompt to analyze", example = "Who weighs more? An Elephant or a Rhino? And what event bigger ones live in Africa?")
            String text
    ) {}

    public record HintResponse(
            @ArraySchema(schema = @Schema(implementation = String.class))
            List<String> hints,
            @Schema(description = "Elapsed time in milliseconds") long elapsedMs,
            @Schema(description = "Echo of the analyzed text") String input
    ) {}
}
