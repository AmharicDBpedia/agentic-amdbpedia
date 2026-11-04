FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=600 \
    UV_CONCURRENT_DOWNLOADS=4

WORKDIR /app

COPY pyproject.toml uv.lock ./
# Flaky-network tolerant sync: a single wheel timeout fails the whole
# resolve, so retry a few times before giving up. The cache mount keeps
# already-downloaded wheels across build attempts, so each retry picks
# up where the last left off. The explicit `exit 1` after the loop is
# load-bearing: without it, a loop whose final iteration ends on
# `sleep 10` would return 0 and ship an empty venv.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    for i in 1 2 3 4 5; do \
        uv sync --frozen --no-dev --no-install-project && exit 0; \
        echo "uv sync attempt $i failed, retrying..." && sleep 10; \
    done; echo "uv sync failed after 5 attempts"; exit 1

COPY . .

# Trim the venv: strip the per-package `tests` tree shipped by the big
# wheels (transformers, sentence_transformers, fastembed, ...).
# Deliberately NOT touching `test`, `testing`, or `examples` -- several
# packages import real public submodules by those exact names
# (`torch.testing` loaded during `torch.autograd`, `gepa.examples` loaded
# from `gepa/__init__.py`), and deleting them wedges every import.
# `|| true` because not every wheel ships a `tests` subtree.
RUN set -eux && \
    find /app/.venv -type d -name tests -prune -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type d -name "__pycache__" -empty -delete 2>/dev/null || true

FROM python:3.11-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/data/.hf-cache \
    # Keep HuggingFace tokenizers single-threaded: the worker-pool fork
    # is exactly the thing that caused the `Broken pipe` wedge the
    # recent fix recovers from. Belt-and-braces: avoid the fork at all.
    TOKENIZERS_PARALLELISM=false \
    # torch/numpy on a CPU-only inference box don't benefit from BLAS
    # thread fanout and the extra threads just contend; one worker is
    # enough for the retrieval query hot-path (~afro-xlmr base, 110 M
    # params on CPU).
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY --from=builder --chown=appuser:appuser /app /app

# /app/data is written on first retrieval request (index cache,
# HF model download). Pre-create and chown so the non-root user can
# write without a startup bind-mount dance.
RUN mkdir -p /app/data/.hf-cache /app/data/.cache && \
    chown -R appuser:appuser /app/data

USER appuser

EXPOSE 8001

# Lightweight readiness probe: GET /v1/coverage returns 200 with an
# empty-corpus JSON body as soon as the HTTP app has booted; it does
# NOT force the embedding model to load, so docker reports healthy
# well before the first real request pays the warm-up cost.
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/v1/coverage', timeout=3).status==200 else 1)"

CMD ["python", "scripts/run_http.py", "--host", "0.0.0.0", "--port", "8001"]
