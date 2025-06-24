#!/bin/bash
# init.sh
# Run original entrypoint
docker-entrypoint.sh postgres &

# Check and create the database and schema manually if needed
psql -U admin -d postgres -c "CREATE DATABASE nuclide;"
psql -U admin -d postgres -c "CREATE DATABASE mlflowdb;"
psql -U admin -d nuclide -c "CREATE SCHEMA IF NOT EXISTS measurements;"
psql -U admin -d nuclide -c "CREATE SCHEMA IF NOT EXISTS nuclide;"
psql -U admin -d nuclide -c "CREATE SCHEMA IF NOT EXISTS meta;"

