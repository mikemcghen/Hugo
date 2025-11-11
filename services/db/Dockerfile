# PostgreSQL with pgvector
# Long-term memory database with vector search

FROM postgres:16

# Install pgvector extension
RUN apt-get update && apt-get install -y \
    postgresql-16-pgvector \
    && rm -rf /var/lib/apt/lists/*

# Copy initialization script
COPY init.sql /docker-entrypoint-initdb.d/

EXPOSE 5432
