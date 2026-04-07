FROM postgres:16

# Install the pg_hint_plan extension for Postgres 16
RUN apt-get update && \
    apt-get install -y --no-install-recommends postgresql-16-pg-hint-plan && \
    rm -rf /var/lib/apt/lists/*