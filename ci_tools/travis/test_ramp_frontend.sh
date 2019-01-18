psql -U postgres -c "CREATE USER mrramp WITH PASSWORD 'mrramp';ALTER USER mrramp WITH SUPERUSER;"
createdb --owner=mrramp databoard_test
python -m smtpd -n -c DebuggingServer localhost:8025 &

pytest -rvsl ramp-frontend --cov=ramp-frontend --cov-report=term-missing
