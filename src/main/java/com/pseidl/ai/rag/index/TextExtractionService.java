package com.pseidl.ai.rag.index;

import static java.util.Objects.requireNonNull;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.StringJoiner;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.tika.metadata.Metadata;
import org.apache.tika.parser.AutoDetectParser;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.sax.BodyContentHandler;
import org.springframework.stereotype.Service;

@Service
public class TextExtractionService {

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

            case "csv":
                return extractCsv(file);

            default:
                String text = Files.readString(file, StandardCharsets.UTF_8);
                return normalize(text);
        }
    }

    // ---- CSV (header-aware) ----
    private String extractCsv(Path file) throws IOException {
        char delim = sniffDelimiter(file);
        CSVFormat fmt = CSVFormat.DEFAULT.builder()
                .setDelimiter(delim)
                .setTrim(true)
                .setIgnoreSurroundingSpaces(true)
                .setQuote('"')
                .setHeader()              // read header from first record
                .setSkipHeaderRecord(true)
                .build();

        StringBuilder out = new StringBuilder(8192);
        try (Reader r = Files.newBufferedReader(file, StandardCharsets.UTF_8);
             CSVParser parser = new CSVParser(r, fmt)) {

            List<String> headers = parser.getHeaderNames();
            int row = 0;
            for (CSVRecord rec : parser) {
                row++;
                StringJoiner sj = new StringJoiner(" | ");
                for (String h : headers) {
                    String v = rec.isMapped(h) ? rec.get(h) : "";
                    sj.add(h + "=" + quoteIfNeeded(v));
                }
                out.append("row ").append(row).append(": ").append(sj).append("\n");
            }
        }
        
        String res = normalize(out.toString());
        //System.err.println(res);
        return res;
    }

    private static String quoteIfNeeded(String v) {
        if (v == null) return "\"\"";
        String s = v.trim();
        // quote if contains whitespace or separators
        if (s.isEmpty() || s.matches(".*[\\s\\|:].*")) {
            return "\"" + s.replace("\"", "\\\"") + "\"";
        }
        return s;
    }

    private static char sniffDelimiter(Path file) throws IOException {
        try (BufferedReader br = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
            String first = br.readLine();
            if (first == null) return ',';
            int cComma = count(first, ',');
            int cSemi  = count(first, ';');
            int cTab   = count(first, '\t');
            int cPipe  = count(first, '|');
            int max = Math.max(Math.max(cComma, cSemi), Math.max(cTab, cPipe));
            if (max == cSemi) return ';';
            if (max == cTab)  return '\t';
            if (max == cPipe) return '|';
            return ','; // default
        }
    }

    private static int count(String s, char ch) {
        int n = 0;
        for (int i = 0; i < s.length(); i++) if (s.charAt(i) == ch) n++;
        return n;
    }

    // ---- Tika for binary docs ----
    private String extractWithTika(Path file) throws IOException {
        try (InputStream is = Files.newInputStream(file)) {
            AutoDetectParser parser = new AutoDetectParser();
            BodyContentHandler handler = new BodyContentHandler(-1);
            Metadata metadata = new Metadata();
            ParseContext ctx = new ParseContext();

            parser.parse(is, handler, metadata, ctx);
            return normalize(handler.toString());
        } catch (Exception e) {
            throw new IOException("Tika parse failed for " + file.getFileName() + ": " + e.getMessage(), e);
        }
    }

    // ---- helpers ----
    private String normalize(String s) {
        if (s == null) return "";
        String t = s.replace("\r\n", "\n").replace('\r', '\n');
        t = t.replaceAll("[ \\t\\u00A0\\f\\u000B]+", " ");
        t = t.replaceAll("\\n{3,}", "\n\n");
        return t.trim();
    }

    private static String extension(Path p) {
        String name = p.getFileName().toString().toLowerCase(Locale.ROOT);
        int dot = name.lastIndexOf('.');
        return (dot >= 0) ? name.substring(dot + 1) : "";
    }
}
