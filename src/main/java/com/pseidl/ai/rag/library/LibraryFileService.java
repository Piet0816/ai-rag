package com.pseidl.ai.rag.library;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.BasicFileAttributes;
import java.time.Instant;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import com.pseidl.ai.rag.config.AppConfig;

/**
 * Minimal file access for the library folder:
 * - List files with a reasonable default set of extensions.
 * - Read a file by relative path with path traversal protection.
 */
@Service
public class LibraryFileService {

    private static final Logger log = LoggerFactory.getLogger(LibraryFileService.class);

    /** Comma-separated default extensions from configuration. */
    @Value("${app.ingest.extensions:txt,md,csv,json,xml,yaml,yml,properties,java,kt,py,js,ts,tsx,sql,gradle,sh,bat}")
    private String defaultExtCsv;

    private final Path root;

    public LibraryFileService(AppConfig appConfig) {
        this.root = Paths.get(appConfig.getLibraryDir()).toAbsolutePath().normalize();
    }

    public Path getRoot() {
        return root;
    }

    /** List files under the library root using the default extension filter. */
    public List<FileEntry> listFiles() throws IOException {
    	Set<String> set = Stream.of(defaultExtCsv.split(",")).collect(Collectors.toSet());
        return listFiles(set);
    }

    /** List files under the library root using a custom extension allowlist (lowercase, no dots). */
    public List<FileEntry> listFiles(Set<String> allowedExtensions) throws IOException {
        try (Stream<Path> stream = Files.walk(root)) {
            return stream
                    .filter(Files::isRegularFile)
                    .filter(p -> allowedExtensions.isEmpty() || allowedExtensions.contains(extensionOf(p)))
                    .map(this::toEntry)
                    .sorted(Comparator.comparing(FileEntry::relativePath))
                    .toList();
        }
    }

    /** Read a file (UTF-8 text) by relative path. Throws if outside root or not a file. */
    public String readFile(String relativePath) throws IOException {
        Path p = safeResolve(relativePath);
        if (!Files.isRegularFile(p)) {
            throw new IOException("Not a regular file: " + relativePath);
        }
        return Files.readString(p);
    }

    /** Read raw bytes by relative path. */
    public byte[] readFileBytes(String relativePath) throws IOException {
        Path p = safeResolve(relativePath);
        if (!Files.isRegularFile(p)) {
            throw new IOException("Not a regular file: " + relativePath);
        }
        return Files.readAllBytes(p);
    }

    private Path safeResolve(String relativePath) throws IOException {
        Path p = root.resolve(relativePath).normalize();
        if (!p.startsWith(root)) {
            throw new IOException("Path escapes library root: " + relativePath);
        }
        return p;
    }

    private FileEntry toEntry(Path p) {
        try {
            BasicFileAttributes a = Files.readAttributes(p, BasicFileAttributes.class);
            String rel = root.relativize(p).toString().replace('\\', '/');
            return new FileEntry(rel, a.size(), a.lastModifiedTime().toInstant());
        } catch (IOException e) {
            log.warn("Failed to read attributes for {}", p, e);
            String rel = root.relativize(p).toString().replace('\\', '/');
            return new FileEntry(rel, -1L, Instant.EPOCH);
        }
    }

    private static String extensionOf(Path p) {
        String name = p.getFileName().toString();
        int dot = name.lastIndexOf('.');
        return dot < 0 ? "" : name.substring(dot + 1).toLowerCase(Locale.ROOT);
    }

    /** Lightweight file descriptor used by listFiles(). */
    public record FileEntry(String relativePath, long size, Instant lastModified) {}
}
