package com.pseidl.ai.rag.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Binds application properties and performs minimal startup initialization.
 * - Maps: app.library-dir, app.ollama.base-url, app.ollama.embedding-model, app.ollama.chat-model
 * - Ensures the library directory exists on startup.
 */
@Configuration
@ConfigurationProperties(prefix = "app")
public class AppConfig {

    private static final Logger log = LoggerFactory.getLogger(AppConfig.class);

    /** Folder for your local document/code library. */
    private String libraryDir = "./library";

    /** Ollama-related settings. */
    private final Ollama ollama = new Ollama();

    public String getLibraryDir() {
        return libraryDir;
    }

    public void setLibraryDir(String libraryDir) {
        this.libraryDir = libraryDir;
    }

    public Ollama getOllama() {
        return ollama;
    }

    public static class Ollama {
    	
        private String baseUrl = "http://localhost:11434";
        private String embeddingModel;
        private String chatModel;

        public String getBaseUrl() {
            return baseUrl;
        }

        public void setBaseUrl(String baseUrl) {
            this.baseUrl = baseUrl;
        }

        public String getEmbeddingModel() {
            return embeddingModel;
        }

        public void setEmbeddingModel(String embeddingModel) {
            this.embeddingModel = embeddingModel;
        }

        public String getChatModel() {
            return chatModel;
        }

        public void setChatModel(String chatModel) {
            this.chatModel = chatModel;
        }
    }

    /**
     * Uses the mapped properties to create the library directory on startup and log effective settings.
     */
    @Bean
    CommandLineRunner initLibrary() {
        return args -> {
            Path lib = Paths.get(libraryDir).toAbsolutePath();
            ensureDirectory(lib);
            log.info("Using library directory: {}", lib);
            log.info("Ollama base URL: {}", ollama.getBaseUrl());
            log.info("Embedding model: {}", ollama.getEmbeddingModel());
            log.info("Chat model: {}", ollama.getChatModel());
        };
    }

    private void ensureDirectory(Path dir) throws IOException {
        if (Files.exists(dir)) {
            if (!Files.isDirectory(dir)) {
                throw new IOException("Configured library path exists but is not a directory: " + dir);
            }
        } else {
            Files.createDirectories(dir);
            log.info("Created library directory at {}", dir);
        }
    }
    
}
