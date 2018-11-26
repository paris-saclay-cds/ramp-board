psql -U postgres -c "CREATE USER ${DATABOARD_USER} WITH PASSWORD '${DATABOARD_PASSWORD}';ALTER USER ${DATABOARD_USER} WITH SUPERUSER;"
createdb --owner=${DATABOARD_USER} databoard_test

pytest -vsl databoard
