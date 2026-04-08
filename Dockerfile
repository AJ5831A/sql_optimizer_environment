# ─────────────────────────────────────────────────────────────────────────────
# Self-contained Dockerfile for Hugging Face Spaces
#
# Bundles Postgres 16 + pg_hint_plan + the OpenEnv FastAPI server in a single
# image so the Space works out of the box (HF Spaces runs one container per
# Space and has no external DB).
#
# For local development prefer `docker-compose up` which uses the split
# two-container layout (sql_optimizer/server/Dockerfile + db.Dockerfile).
# ─────────────────────────────────────────────────────────────────────────────

FROM postgres:16

# ── System deps: Python + build tools + pg_hint_plan ────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        postgresql-16-pg-hint-plan \
        libpq-dev gcc curl && \
    rm -rf /var/lib/apt/lists/*

# ── Postgres bootstrap ───────────────────────────────────────────────────────
ENV POSTGRES_USER=sqlopt \
    POSTGRES_PASSWORD=sqlopt \
    POSTGRES_DB=sqlopt \
    PGDATA=/var/lib/postgresql/data \
    DATABASE_URL=postgresql://sqlopt:sqlopt@localhost:5432/sqlopt

# Bake the sample schema into the initdb hook so it runs on first boot
COPY sample_db_init.sql /docker-entrypoint-initdb.d/01_init.sql

# ── Python app ───────────────────────────────────────────────────────────────
WORKDIR /app

COPY pyproject.toml ./
COPY uv.lock* ./
COPY sql_optimizer ./sql_optimizer
COPY server ./server
COPY client.py models.py __init__.py ./

RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir \
        "openenv-core[core]>=0.2.2" \
        "psycopg2-binary>=2.9.0" \
        "sqlglot>=23.0.0" \
        "fastapi>=0.100.0" \
        "uvicorn[standard]>=0.20.0" \
        "pydantic>=2.0.0"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH" \
    PYTHONUNBUFFERED=1

# ── Startup: boot Postgres, then the OpenEnv server ─────────────────────────
COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/local/bin/start.sh"]
