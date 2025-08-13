package com.pseidl.ai.rag.index;

import static java.util.Objects.requireNonNull;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

import org.apache.tika.exception.TikaException;
import org.apache.tika.metadata.Metadata;
import org.apache.tika.parser.AutoDetectParser;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.sax.BodyContentHandler;
import org.springframework.stereotype.Service;


/**
 * Small utility to extract text from files.
 * - Uses Apache Tika for binary docs (pdf, doc, docx, …)
 * - Falls back to UTF-8 read for plain text formats (txt, md, csv, json, xml, yaml, properties, code files)
 *
 * Usage (next step): inject into LibraryIngestionService and call extract(filePath) to get text.
 */
@Service
public class TextExtractionService {

    /**
     * Extracts text content from the given file.
     * @param file absolute path to the file
     */
    public String extract(Path file) throws IOException {
        requireNonNull(file, "file");
        String ext = extension(file);

        switch (ext) {
            case "pdf":
            case "doc":
            case "docx":
            case "rtf":
            case "odt":
            case "ppt":
            case "pptx":
            case "xls":
            case "xlsx":
                return extractWithTika(file);

            default:
                // Plain text family: read as UTF-8
                // (txt, md, csv, json, xml, yaml/yml, properties, code files, etc.)
                String text = Files.readString(file, StandardCharsets.UTF_8);
                return normalize(text);
        }
    }

    // --- internals ---

    private String extractWithTika(Path file) throws IOException {
        try (InputStream is = Files.newInputStream(file)) {
            AutoDetectParser parser = new AutoDetectParser();
            // -1 => no character limit (let ingestion chunking handle size)
            BodyContentHandler handler = new BodyContentHandler(-1);
            Metadata metadata = new Metadata();
            ParseContext ctx = new ParseContext();

            parser.parse(is, handler, metadata, ctx);
            String text = handler.toString();
            return normalize(text);
        } catch (Exception e) {
            throw new IOException("Tika parse failed for " + file.getFileName() + ": " + e.getMessage(), e);
        }
    }

    /** Light normalization for indexing: unify newlines, collapse runs of spaces, keep paragraph breaks. */
    private String normalize(String s) {
        if (s == null) return "";
        String t = s.replace("\r\n", "\n").replace('\r', '\n');
        // collapse runs of spaces/tabs but keep newlines
        t = t.replaceAll("[ \\t\\u00A0\\f\\u000B]+", " ");
        // squeeze huge blank gaps to max 2 newlines
        t = t.replaceAll("\\n{3,}", "\n\n");
        return t.trim();
    }

    private static String extension(Path p) {
        String name = p.getFileName().toString().toLowerCase(Locale.ROOT);
        int dot = name.lastIndexOf('.');
        return (dot >= 0) ? name.substring(dot + 1) : "";
    }
}
