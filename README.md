# Mneme Memory Service

[![Build & Release](https://github.com/wuai1024/mneme-memory/actions/workflows/release.yml/badge.svg)](https://github.com/wuai1024/mneme-memory/actions/workflows/release.yml)
[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/wuai1024/mneme-memory?sort=semver)](https://github.com/wuai1024/mneme-memory/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> **[中文文档](./README_CN.md)**

**Semantic persistent memory service** — A lightweight solution for building long-term memory for AI Agents.

Supports storing facts in **Subject → Predicate → Object** triplets, combining sentence-transformers vector embeddings with SQLite for parallel semantic search and keyword retrieval.

---

## Features

- **Semantic Search** — Vector similarity with `all-MiniLM-L6-v2` model
- **Keyword Search** — Full-text search via SQLite FTS5
- **Hybrid Search** — Parallel semantic + keyword with deduplication
- **Triple Storage** — `Subject / Predicate / Object` structured memory
- **Session Summaries** — Conversation-level summary storage
- **Batch Operations** — Batch create/delete for facts and summaries
- **Data Export/Import** — Full data backup and restore
- **Metrics** — Prometheus-compatible metrics endpoint
- **API Key Auth** — Stateless authentication for all endpoints
- **Single SQLite File** — No external database, just one `.db` file
- **Lightweight** — CPU-only, x86_64/arm64, image < 300MB

---

## Quick Start

### 1. Start Service

```bash
docker run -d \
  --name mneme-memory \
  -p 33333:33333 \
  -v $(pwd)/data:/data \
  -e MEMORY_API_KEY="your-secret-key-here" \
  ghcr.io/wuai1024/mneme-memory:latest
```

> `MEMORY_API_KEY` is required. All API requests must include `X-API-Key` header.

### 2. Verify

```bash
curl http://localhost:33333/health
# {"status":"ok","embedding_model":"all-MiniLM-L6-v2","embedding_dimension":384,"version":"1.1.0"}
```

### 3. Store a Fact

```bash
curl -X POST http://localhost:33333/facts \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{"subject":"Alice","predicate":"lives in","object":"Shanghai"}'
```

### 4. Semantic Search

```bash
curl -X POST http://localhost:33333/search \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{"query":"Where does Alice live?","top_k":3}'
```

---

## API Reference

### Core Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `GET` | `/health` | Health check | ❌ |
| `POST` | `/facts` | Create fact | ✅ |
| `POST` | `/facts/batch` | Batch create facts | ✅ |
| `GET` | `/facts` | List facts | ✅ |
| `GET` | `/facts/{id}` | Get fact | ✅ |
| `PUT` | `/facts/{id}` | Update fact | ✅ |
| `DELETE` | `/facts/{id}` | Delete fact | ✅ |
| `DELETE` | `/facts/batch` | Batch delete facts | ✅ |
| `POST` | `/summaries` | Create summary | ✅ |
| `POST` | `/summaries/batch` | Batch create summaries | ✅ |
| `GET` | `/summaries` | List summaries | ✅ |
| `DELETE` | `/summaries/{id}` | Delete summary | ✅ |
| `POST` | `/search` | Search | ✅ |

### Data Management

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `GET` | `/export` | Export all data | ✅ |
| `POST` | `/import` | Import data | ✅ |
| `GET` | `/metrics` | Get metrics | ❌ |
| `POST` | `/maintenance/vacuum` | Vacuum database | ✅ |

### Authentication

All authenticated requests must include:

```
X-API-Key: your-secret-key-here
```

Missing or wrong key returns `401 Unauthorized`.

### Search Modes

```json
{
  "query": "Where does Alice live?",
  "top_k": 3,
  "mode": "semantic"  // semantic | keyword | hybrid
}
```

- **semantic** (default): Pure vector similarity search
- **keyword**: Pure FTS5 full-text search
- **hybrid**: Combined semantic + keyword with deduplication

### Batch Operations

```bash
# Batch create
curl -X POST http://localhost:33333/facts/batch \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "facts": [
      {"subject":"Alice","predicate":"lives in","object":"Shanghai"},
      {"subject":"Bob","predicate":"lives in","object":"Beijing"}
    ]
  }'

# Batch delete
curl -X DELETE http://localhost:33333/facts/batch \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{"ids":["id1","id2","id3"]}'
```

---

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MEMORY_API_KEY` | **Required** | API authentication key |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model name |
| `DATA_DIR` | `/data` | SQLite database directory |
| `PORT` | `33333` | Service port |
| `SKIP_AUTH` | `false` | Skip auth (dev only) |
| `CORS_ORIGINS` | `*` | CORS allowed origins |
| `MODEL_CACHE` | `~/.cache/huggingface/` | Model cache directory |

---

## Project Structure

```
.
├── app/
│   ├── main.py          # FastAPI entry point
│   ├── database.py      # SQLite + vector search (numpy optimized)
│   ├── embedding.py     # sentence-transformers wrapper
│   └── models.py        # Pydantic models
├── Dockerfile           # CPU-only image (x86_64 / arm64)
├── requirements.txt     # Python dependencies
├── docker-compose.yml   # Docker Compose config
├── .env.example         # Environment template
└── README.md
```

---

## Local Development

```bash
# Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Start service (dev mode with hot reload)
MEMORY_API_KEY=your-secret SKIP_AUTH=true uvicorn app.main:app --reload --port 33333

# Run tests
python test_api.py
```

> First startup downloads the embedding model (~90MB), cached to `~/.cache/huggingface/`.

---

## Performance

v1.1.0 optimizations:

1. **Connection Pool**: Thread-local connection pool
2. **numpy Vectors**: 5-10x faster cosine similarity
3. **Batch Operations**: Reduce network round-trips
4. **WAL Mode**: Better SQLite concurrency
5. **Index Optimization**: Indexed query fields

---

## Security

- **Set strong `MEMORY_API_KEY`** (32+ bytes recommended)
- `SKIP_AUTH=true` only for local development
- Database file contains all memories, control file permissions
- CORS defaults to `*`, restrict via `CORS_ORIGINS` in production
- API key comparison uses `hmac.compare_digest` to prevent timing attacks

---

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
