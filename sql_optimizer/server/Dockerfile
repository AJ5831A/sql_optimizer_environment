# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile
#
# Uses the official OpenEnv base image.
# The base image provides: uv, fastapi, uvicorn, openenv-core.
#
# Build via OpenEnv CLI (recommended):
#   openenv build
#
# Or manually from repo root:
#   docker build -t sql_optimizer:latest -f sql_optimizer/server/Dockerfile .
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# libpq-dev + gcc needed to compile psycopg2 from source
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency files first — better layer caching
COPY pyproject.toml ./
COPY uv.lock* ./

# Install deps (without the package itself first)
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

# Copy application code
COPY . .

# Install the package itself
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app       /app/env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# WORKERS is configurable via docker-compose env or openenv.yaml
CMD ["sh", "-c", "cd /app/env && uvicorn sql_optimizer.server.app:app \
     --host 0.0.0.0 \
     --port 8000 \
     --workers ${WORKERS:-4}"]