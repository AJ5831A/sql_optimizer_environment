#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh — bring up Postgres + the OpenEnv FastAPI server in one container.
#
# Used by the self-contained HF Spaces Dockerfile at the repo root. Local dev
# with docker-compose uses the split two-container layout instead.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PGDATA="${PGDATA:-/var/lib/postgresql/data}"

# HF Spaces mounts /data with restrictive perms; make sure postgres owns PGDATA
mkdir -p "$PGDATA"
chown -R postgres:postgres "$PGDATA"
chmod 700 "$PGDATA"

# First boot: initdb + run /docker-entrypoint-initdb.d/*.sql
if [ -z "$(ls -A "$PGDATA" 2>/dev/null || true)" ]; then
    echo "[start] Bootstrapping Postgres data dir..."
    # Delegate to the official postgres entrypoint in background, then stop it
    # cleanly once init is done. We run it in "single init" mode via pg_ctl.
    su postgres -c "/usr/lib/postgresql/16/bin/initdb -D $PGDATA -U sqlopt --pwfile=<(echo sqlopt)"

    # Start postgres briefly to run the init SQL
    su postgres -c "/usr/lib/postgresql/16/bin/pg_ctl -D $PGDATA -o '-c listen_addresses=localhost -c shared_preload_libraries=pg_hint_plan' -w start"

    # Create the database (initdb made the sqlopt role; now make the DB)
    su postgres -c "psql -U sqlopt -d postgres -c \"CREATE DATABASE sqlopt OWNER sqlopt;\""

    # Run the baked init script against the fresh DB
    if [ -f /docker-entrypoint-initdb.d/01_init.sql ]; then
        echo "[start] Loading sample schema..."
        su postgres -c "psql -U sqlopt -d sqlopt -f /docker-entrypoint-initdb.d/01_init.sql" || true
    fi

    su postgres -c "/usr/lib/postgresql/16/bin/pg_ctl -D $PGDATA -m fast -w stop"
fi

# Normal start — listen on localhost only; the Python app connects over loopback
echo "[start] Starting Postgres..."
su postgres -c "/usr/lib/postgresql/16/bin/postgres -D $PGDATA -c listen_addresses=localhost -c shared_preload_libraries=pg_hint_plan" &
PG_PID=$!

# Wait for Postgres to accept connections
echo "[start] Waiting for Postgres..."
for i in $(seq 1 30); do
    if pg_isready -h localhost -U sqlopt -d sqlopt >/dev/null 2>&1; then
        echo "[start] Postgres is ready."
        break
    fi
    sleep 1
done

# Graceful shutdown handler
trap 'echo "[start] Shutting down..."; kill -TERM $PG_PID 2>/dev/null || true; wait $PG_PID 2>/dev/null || true; exit 0' TERM INT

echo "[start] Starting OpenEnv FastAPI server on :8000..."
exec uvicorn sql_optimizer.server.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers "${WORKERS:-1}"
