# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile for Hugging Face Spaces
#
# Runs the OpenEnv FastAPI server. POST /reset succeeds even without a
# database: the env returns a degraded observation with a clear message
# explaining how to supply `db_url`. Real usage: pass a reachable Postgres
# connection string via reset(db_url=...) to unlock full optimization.
#
# For local development with a real Postgres, use `docker-compose up`.
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# libpq for psycopg2, curl for healthcheck, gcc for wheel fallbacks
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq-dev gcc curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "openenv-core[core]>=0.2.2" \
        "psycopg2-binary>=2.9.0" \
        "sqlglot>=23.0.0" \
        "fastapi>=0.100.0" \
        "uvicorn[standard]>=0.20.0" \
        "pydantic>=2.0.0"

# Copy application code
COPY pyproject.toml ./
COPY sql_optimizer ./sql_optimizer
COPY server ./server
COPY client.py models.py __init__.py ./

ENV PYTHONPATH="/app:$PYTHONPATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "sql_optimizer.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
