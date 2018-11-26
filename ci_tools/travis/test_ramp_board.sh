export DATABOARD_TEST=True
# export DATABOARD_DB_URL=postgresql://mrramp:mrramp@localhost/databoard_test
export DATABOARD_DB_URL_TEST=postgresql://mrramp:mrramp@localhost/databoard_test

# mkdir postgres_dbs
# initdb postgres_dbs
# pg_ctl -D postgres_dbs -l postgres_dbs/logfile start
psql -U postgres -c "CREATE USER mrramp WITH PASSWORD 'mrramp';ALTER USER mrramp WITH SUPERUSER;"
createdb --owner=mrramp databoard_test
# create a user and set a password
# createuser --pwprompt mrramp
# create the database
pytest -vsl databoard
