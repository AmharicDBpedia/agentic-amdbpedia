FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .

FROM python:3.11-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/data/.hf-cache

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY --from=builder --chown=appuser:appuser /app /app

USER appuser

EXPOSE 8001

CMD ["python", "scripts/run_http.py", "--host", "0.0.0.0", "--port", "8001"]
