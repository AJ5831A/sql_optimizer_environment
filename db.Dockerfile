FROM postgres:16

# Install the pg_hint_plan extension for Postgres 16
RUN apt-get update && \
    apt-get install -y --no-install-recommends postgresql-16-pg-hint-plan && \
    rm -rf /var/lib/apt/lists/*

# Bake the init script into the image rather than bind-mounting it.
# Bind-mounting a single file from macOS into Postgres triggers a VirtioFS
# "Resource deadlock avoided" error in Docker Desktop.
COPY sample_db_init.sql /docker-entrypoint-initdb.d/01_init.sql