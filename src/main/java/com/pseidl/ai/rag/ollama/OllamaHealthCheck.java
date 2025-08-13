package com.pseidl.ai.rag.ollama;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientResponseException;

import com.fasterxml.jackson.databind.JsonNode;
import com.pseidl.ai.rag.config.AppConfig;

/**
 * Verifies Ollama availability and configured models on application startup.
 * Robustly matches model names with/without tags (e.g., ":latest", ":8b").
 *
 * Optional config (application.properties or env/system props):
 *   app.ollama.health.required=true
 *   app.ollama.health.require-models=true
 *   app.ollama.health.max-log-models=100
 */
@Component
public class OllamaHealthCheck implements CommandLineRunner {

    private static final Logger log = LoggerFactory.getLogger(OllamaHealthCheck.class);

    private final RestClient http;
    private final AppConfig app;

    private final boolean required;
    private final boolean requireModels;
    private final int maxLogModels;

    public OllamaHealthCheck(RestClient.Builder builder, AppConfig app) {
        this.app = app;
        this.http = builder.baseUrl(app.getOllama().getBaseUrl()).build();

        this.required      = getBool("app.ollama.health.required", true);
        this.requireModels = getBool("app.ollama.health.require-models", true);
        this.maxLogModels  = getInt("app.ollama.health.max-log-models", 100);
    }

    @Override
    public void run(String... args) {
        final String baseUrl    = app.getOllama().getBaseUrl();
        final String embedModel = app.getOllama().getEmbeddingModel();
        final String chatModel  = app.getOllama().getChatModel();

        log.info("Checking Ollama at {}", baseUrl);

        List<ModelInfo> models;
        try {
            models = fetchInstalledModels();
        } catch (Exception e) {
            String msg = "Ollama not reachable at " + baseUrl + " (" + e.getClass().getSimpleName() + ": " + e.getMessage() + ")";
            if (required) throw new IllegalStateException(msg, e);
            log.warn("{} — continuing because app.ollama.health.required=false", msg);
            return;
        }

        // Build canonical set: name, model, and both without tags; all lowercased
        Set<String> canonical = new LinkedHashSet<>();
        for (ModelInfo mi : models) {
            addCanonical(canonical, mi.name());
            addCanonical(canonical, mi.modelTag());
        }

        // Log installed (pretty) + canonical (debug)
        List<String> pretty = models.stream().map(ModelInfo::pretty).sorted().toList();
        if (!pretty.isEmpty()) {
            List<String> head = pretty.subList(0, Math.min(pretty.size(), maxLogModels));
            log.info("Installed models ({} total, showing up to {}): {}", pretty.size(), maxLogModels, head);
        } else {
            log.info("No models reported by /api/tags");
        }
        if (log.isDebugEnabled()) {
            log.debug("Canonical model names (lowercase, tagless variants included): {}", canonical);
        }

        boolean embedOk = isInstalled(embedModel, canonical);
        boolean chatOk  = isInstalled(chatModel, canonical);

        log.info("Configured embedding model: {} [{}]", safe(embedModel), embedOk ? "OK" : "MISSING");
        log.info("Configured chat model     : {} [{}]", safe(chatModel),  chatOk  ? "OK" : "MISSING");

        if (requireModels && (!embedOk || !chatOk)) {
            StringBuilder sb = new StringBuilder("Required Ollama models not installed: ");
            if (!embedOk) sb.append("[embedding=").append(safe(embedModel)).append("] ");
            if (!chatOk)  sb.append("[chat=").append(safe(chatModel)).append("]");
            sb.append(". Try: ");
            if (!embedOk && embedModel != null) sb.append("`ollama pull ").append(embedModel).append("` ");
            if (!chatOk  && chatModel  != null) sb.append("`ollama pull ").append(chatModel).append("`");
            throw new IllegalStateException(sb.toString().trim());
        }

        // Optional extra: log version if available
        try {
            String version = fetchVersion();
            if (version != null) log.info("Ollama version: {}", version);
        } catch (Exception ignore) {}
    }

    // --- helpers ---

    private static void addCanonical(Set<String> set, String name) {
        if (name == null || name.isBlank()) return;
        String n = name.trim();
        String lc = n.toLowerCase(Locale.ROOT);
        set.add(lc);
        int colon = lc.indexOf(':');
        if (colon > 0) {
            set.add(lc.substring(0, colon)); // tagless
        }
        // also add any path tail after a slash without tag (e.g. "user/model:tag" -> "model")
        int slash = lc.lastIndexOf('/');
        if (slash >= 0) {
            String tail = lc.substring(slash + 1);
            set.add(tail);
            int colon2 = tail.indexOf(':');
            if (colon2 > 0) set.add(tail.substring(0, colon2));
        }
    }

    private static boolean isInstalled(String configured, Set<String> canonical) {
        if (configured == null || configured.isBlank()) return false;
        String c = configured.trim();
        String lc = c.toLowerCase(Locale.ROOT);
        if (canonical.contains(lc)) return true;
        int colon = lc.indexOf(':');
        if (colon > 0 && canonical.contains(lc.substring(0, colon))) return true; // configured has tag; store has tagless
        // also accept default ":latest" resolution
        if (!lc.contains(":") && canonical.contains(lc + ":latest")) return true;
        return false;
    }

    private List<ModelInfo> fetchInstalledModels() {
        try {
            JsonNode node = http.get()
                    .uri("/api/tags")
                    .accept(MediaType.APPLICATION_JSON)
                    .retrieve()
                    .body(JsonNode.class);
            return parseTags(node);
        } catch (RestClientResponseException e) {
            // Some versions accept POST {} instead of GET
            JsonNode node = http.post()
                    .uri("/api/tags")
                    .contentType(MediaType.APPLICATION_JSON)
                    .body("{}")
                    .retrieve()
                    .body(JsonNode.class);
            return parseTags(node);
        }
    }

    private List<ModelInfo> parseTags(JsonNode node) {
        if (node == null) return List.of();
        List<ModelInfo> out = new ArrayList<>();
        if (node.has("models") && node.get("models").isArray()) {
            for (JsonNode m : node.get("models")) {
                String name  = m.has("name")  ? m.get("name").asText()  : null;        // often "llama3", "mxbai-embed-large"
                String model = m.has("model") ? m.get("model").asText() : null;        // often "llama3:8b", "...:latest"
                String param = (m.has("details") && m.get("details").has("parameter_size"))
                        ? m.get("details").get("parameter_size").asText() : null;
                String quant = (m.has("details") && m.get("details").has("quantization_level"))
                        ? m.get("details").get("quantization_level").asText() : null;
                out.add(new ModelInfo(name, model, param, quant));
            }
        }
        return out;
    }

    private String fetchVersion() {
        try {
            JsonNode node = http.get()
                    .uri("/api/version")
                    .accept(MediaType.APPLICATION_JSON)
                    .retrieve()
                    .body(JsonNode.class);
            if (node != null && node.has("version")) return node.get("version").asText();
        } catch (Exception ignore) {}
        return null;
    }

    private static String safe(String s) { return (s == null || s.isBlank()) ? "-" : s; }

    private static boolean getBool(String key, boolean def) {
        String v = System.getProperty(key);
        if (v == null) v = System.getenv(key.replace('.', '_').toUpperCase(Locale.ROOT));
        if (v == null) return def;
        return v.equalsIgnoreCase("true") || v.equalsIgnoreCase("1") || v.equalsIgnoreCase("yes");
    }

    private static int getInt(String key, int def) {
        String v = System.getProperty(key);
        if (v == null) v = System.getenv(key.replace('.', '_').toUpperCase(Locale.ROOT));
        if (v == null) return def;
        try { return Integer.parseInt(v.trim()); } catch (Exception e) { return def; }
    }

    private record ModelInfo(String name, String modelTag, String parameterSize, String quantization) {
        String pretty() {
            List<String> extras = new ArrayList<>();
            if (modelTag != null && !Objects.equals(modelTag, name)) extras.add(modelTag);
            if (parameterSize != null) extras.add(parameterSize);
            if (quantization != null) extras.add(quantization);
            return extras.isEmpty() ? name : name + " (" + String.join(", ", extras) + ")";
        }
    }
}
