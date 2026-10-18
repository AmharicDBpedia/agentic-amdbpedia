# Backend API image: hybrid RAG pipeline + review-queue HTTP server.
# Uses the official uv image so the torch-CPU index pin in pyproject.toml
# (and the uv.lock resolution) is honored exactly as in local dev.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    # Persist the downloaded embedding model + built index on a volume so a
    # restart doesn't re-download/re-embed the ~2,900-property corpus.
    HF_HOME=/app/data/.hf-cache

WORKDIR /app

# Dependency layer first for caching -- only re-runs when the lock changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Application source.
COPY config.py errors.py logging_config.py ./
COPY db ./db
COPY rag ./rag
COPY mcp_server ./mcp_server
COPY scripts ./scripts
COPY data ./data

EXPOSE 8001

# 0.0.0.0 so the server is reachable from outside the container.
CMD ["python", "scripts/run_http.py", "--host", "0.0.0.0", "--port", "8001"]
