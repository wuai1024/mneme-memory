# syntax=docker/dockerfile:1
FROM python:3.12-slim

LABEL org.opencontainers.image.title="Mneme Memory Service"
LABEL org.opencontainers.image.description="Semantic long-term memory service for AI agents"
LABEL org.opencontainers.image.source="https://github.com/wuai1024/mneme-memory"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir "sentence-transformers>=3.0" fastapi uvicorn pydantic

# Pre-download embedding model at build time
RUN python3 -c " \
    from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/data/model_cache')"

COPY app/ ./app/

ENV DATA_DIR=/data
ENV MODEL_CACHE=/data/model_cache
ENV EMBEDDING_MODEL=all-MiniLM-L6-v2
ENV PORT=33333
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 33333

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:33333/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "33333"]
