ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="Mneme Memory Service"
LABEL org.opencontainers.image.description="Semantic long-term memory service for AI agents"
LABEL org.opencontainers.image.source="https://github.com/wuai1024/mneme-memory"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && rm -rf /var/lib/apt/lists/*

# CPU: torch from PyTorch CPU-only wheel index
# GPU: torch from default PyPI (CUDA-enabled by default)
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG CUDA_DEPS=""
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX} ${CUDA_DEPS}

# Install remaining deps
RUN pip install --no-cache-dir \
        "sentence-transformers>=3.0" \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.30" \
        "pydantic>=2.0"

COPY app/ ./app/

ENV DATA_DIR=/data
ENV MODEL_CACHE=/data/model_cache
ENV EMBEDDING_MODEL=all-MiniLM-L6-v2
ENV PORT=33333

EXPOSE 33333

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:33333/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "33333"]
