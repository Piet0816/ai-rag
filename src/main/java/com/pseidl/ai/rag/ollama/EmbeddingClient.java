package com.pseidl.ai.rag.ollama;

import com.fasterxml.jackson.databind.JsonNode;
import com.pseidl.ai.rag.config.AppConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientResponseException;

import java.util.HashMap;
import java.util.Map;

/**
 * Minimal client for Ollama embeddings.
 * - Prefers POST /api/embed with body: { "model": "...", "input": "..." }
 * - Falls back to POST /api/embeddings with body: { "model": "...", "prompt": "..." }
 * - Returns a single embedding vector for one input string.
 */
@Service
public class EmbeddingClient {

    private static final Logger log = LoggerFactory.getLogger(EmbeddingClient.class);

    private final RestClient http;
    private final String model;

    public EmbeddingClient(RestClient.Builder builder, AppConfig app) {
        this.http = builder
                .baseUrl(app.getOllama().getBaseUrl())
                .build();
        this.model = app.getOllama().getEmbeddingModel();
        if (this.model == null || this.model.isBlank()) {
            log.warn("No embedding model configured (app.ollama.embedding-model is empty)");
        } else {
            log.info("EmbeddingClient using model: {}", this.model);
        }
    }

    /**
     * Create an embedding for the given text.
     * @param text input text
     * @return embedding vector (float[])
     * @throws IllegalStateException if no model configured or the request fails
     */
    public float[] embed(String text) {
        if (model == null || model.isBlank()) {
            throw new IllegalStateException("Embedding model is not configured (app.ollama.embedding-model)");
        }
        // Try /api/embed first
        try {
            Map<String, Object> body = new HashMap<>();
            body.put("model", model);
            body.put("input", text);

            JsonNode node = http.post()
                    .uri("/api/embed")
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(body)
                    .retrieve()
                    .body(JsonNode.class);

            float[] vec = extractEmbedding(node);
            if (vec != null) return vec;

            // If response shape is unexpected, fall back
            log.debug("Unexpected /api/embed response shape, attempting /api/embeddings fallback");
            return embedFallback(text);

        } catch (RestClientResponseException e) {
            // Known endpoint differences between Ollama versions -> try fallback endpoint
            log.debug("Primary /api/embed failed with {} {}, trying /api/embeddings",
                    e.getRawStatusCode(), e.getStatusText());
            return embedFallback(text);
        } catch (Exception e) {
            throw new IllegalStateException("Embedding request failed: " + e.getMessage(), e);
        }
    }

    private float[] embedFallback(String text) {
        try {
            Map<String, Object> body = new HashMap<>();
            body.put("model", model);
            body.put("prompt", text);

            JsonNode node = http.post()
                    .uri("/api/embeddings")
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(body)
                    .retrieve()
                    .body(JsonNode.class);

            float[] vec = extractEmbedding(node);
            if (vec != null) return vec;

            throw new IllegalStateException("Could not parse embedding from /api/embeddings response");

        } catch (Exception e) {
            throw new IllegalStateException("Embedding fallback request failed: " + e.getMessage(), e);
        }
    }

    /**
     * Extracts an embedding from either:
     *  - {"embeddings":[...]} (single array) or {"embeddings":[[...]]} (batch, we take first)
     *  - {"embedding":[...]} (singular)
     */
    private static float[] extractEmbedding(JsonNode root) {
        if (root == null) return null;

        // Newer docs often show "embeddings"
        if (root.has("embeddings")) {
            JsonNode e = root.get("embeddings");
            if (e.isArray()) {
                if (!e.isEmpty() && e.get(0).isArray()) {
                    return toFloatArray(e.get(0));
                }
                return toFloatArray(e);
            }
        }
        // Some responses use singular "embedding"
        if (root.has("embedding")) {
            JsonNode e = root.get("embedding");
            if (e.isArray()) {
                return toFloatArray(e);
            }
        }
        return null;
    }

    private static float[] toFloatArray(JsonNode arr) {
        int n = arr.size();
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (float) arr.get(i).asDouble();
        }
        return out;
    }
}
