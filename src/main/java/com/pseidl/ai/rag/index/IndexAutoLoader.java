package com.pseidl.ai.rag.index;

import com.pseidl.ai.rag.persistence.IndexFileStore;
import com.pseidl.ai.rag.persistence.IndexFileStore.LoadResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

/**
 * Loads the on-disk vector store into the in-memory index on startup.
 * Configurable via application.properties:
 *   app.index.auto-load=true
 *   app.index.load.clear=true
 *   app.index.load.batch-size=200
 *   app.index.load.log-every=100
 */
@Component
public class IndexAutoLoader implements CommandLineRunner {

    private static final Logger log = LoggerFactory.getLogger(IndexAutoLoader.class);

    private final InMemoryVectorIndex index;
    private final IndexFileStore store;

    @Value("${app.index.auto-load:true}")
    private boolean autoLoad;

    @Value("${app.index.load.clear:true}")
    private boolean clearBeforeLoad;

    @Value("${app.index.load.batch-size:200}")
    private int batchSize;

    @Value("${app.index.load.log-every:100}")
    private int logEvery;

    public IndexAutoLoader(InMemoryVectorIndex index, IndexFileStore store) {
        this.index = index;
        this.store = store;
    }

    @Override
    public void run(String... args) {
        if (!autoLoad) {
            log.info("Index auto-load disabled (app.index.auto-load=false). Skipping.");
            return;
        }

        try {
            if (clearBeforeLoad) {
                @SuppressWarnings("unchecked")
                List<String> sources = (List<String>) index.info().getOrDefault("sources", List.of());
                for (String s : sources) {
                    index.removeSource(s);
                }
                log.info("Cleared in-memory index before load ({} sources removed).", sources.size());
            }

            long t0 = System.nanoTime();
            LoadResult r = store.loadIntoIndex(index, Math.max(1, batchSize), Math.max(1, logEvery));
            Map<String, Object> info = index.info();
            int count = ((Number) info.getOrDefault("count", 0)).intValue();
            Integer dim = info.containsKey("dimension") ? ((Number) info.get("dimension")).intValue() : null;

            double sec = (System.nanoTime() - t0) / 1_000_000_000.0;
            log.info("Auto-load complete from {}: loaded {} records in {}s (in-memory chunks={}, dim={})",
                    r.path(), r.records(), String.format("%.2f", sec), count, dim);
        } catch (Exception e) {
            log.warn("Index auto-load failed: {}", e.toString());
        }
    }
}
