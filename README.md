# RAG Spring Boot + Ollama (Simple README)

A tiny Retrieval-Augmented Generation (RAG) app in Spring Boot.  
It ingests files from a **library/** folder, builds embeddings with Ollama, and lets you chat with your documents (web UI included).

## Requirements
- Java 17+ (21 recommended)
- Maven 3.9+
- Ollama running locally at `http://localhost:11434`
- Models (pick one embed + one chat):
  ```bash
  # Embedding
  ollama pull mxbai-embed-large        # good default (English)
  # Chat (choose one that fits 8 GB VRAM)
  ollama pull llama3.1:8b-instruct
  # or: qwen2.5:7b-instruct / mistral:7b-instruct / gemma2:9b
  ```

## Configure
Edit `src/main/resources/application.properties` (minimal example):
```properties
server.port=8017
spring.application.name=rag

app.library-dir=./library
app.ollama.base-url=http://localhost:11434
app.ollama.embedding-model=mxbai-embed-large
app.ollama.chat-model=llama3.1:8b-instruct

# Retrieval
app.retrieval.top-k=6
app.retrieval.max-context-chars=6000
app.retrieval.mmr.enabled=true
app.retrieval.mmr.lambda=0.5
app.retrieval.mmr.overfetch=24

# Ingestion: which extensions to include recursively
app.ingest.extensions=txt,md,csv,json,xml,yaml,yml,properties,java,kt,py,js,ts,tsx,sql,gradle,sh,bat,pdf,doc,docx
```

## Run
```bash
mvn spring-boot:run
```
Open:
- Swagger: **http://localhost:8017/swagger-ui.html**
- Chat UI: **http://localhost:8017/chat/index.html**

## Ingest your files
Put documents into `./library/` (subfolders ok). Supported: **txt, md, csv, pdf, doc, docx** (and common code/data files).
- From UI: **Ingest all** button.
- Or HTTP:
  ```bash
  curl -X POST "http://localhost:8017/api/ingest/all"
  # One file:
  curl -X POST "http://localhost:8017/api/ingest/file?path=people_food.txt"
  ```

### CSV specifics
- CSV is parsed with headers and emitted as `key=value` pairs.
- **Row-based chunking**: retrieval targets individual rows.

## Chat
- Messages stream live; **Stop** button cancels mid-reply.
- Click **Show context** to see the exact text sent to the model.
- `think` levels (FAST…MAX) control length/creativity.
