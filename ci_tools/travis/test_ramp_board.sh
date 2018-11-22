if [[ "$PYTHON_VERSION" == 2.7 ]]; then
    psql -U postgres -c "CREATE USER mrramp WITH PASSWORD 'mrramp';ALTER USER mrramp WITH SUPERUSER;"
    createdb --owner=mrramp databoard_test
    export DATABOARD_TEST=True
    export DATABOARD_DB_URL=postgresql://mrramp:mrramp@localhost/databoard_test
    export DATABOARD_DB_URL_TEST=postgresql://mrramp:mrramp@localhost/databoard_test
    cp databoard/config_local.py databoard/config.py
    make test
fi
