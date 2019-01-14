psql -U postgres -c "CREATE USER mrramp WITH PASSWORD 'mrramp';ALTER USER mrramp WITH SUPERUSER;"
createdb --owner=mrramp databoard_test

pytest -rvsl ramp-frontend --cov=frontend --cov-report=term-missing
