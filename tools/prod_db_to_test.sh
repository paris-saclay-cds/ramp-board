#!/bin/sh    
# Copy databoard files and database from production sever to test server.
# This script has to be run from the test server.
# The script assumes that:
# - sciencefs disk (used to backup prod server) is mounted 
# - migration files are up-to-date on test server
# on the test server (in backup_path)
# The current test database is removed and replace by a copy of the prod database

# db dump to be used to recreate the test database
prod_db_dump=databoard_blabla.dump
# databoard files and code directories on the test server
databoard_path=/mnt/ramp_data/datacamp/databoard
code_path=/mnt/ramp_data/code/databoard
# backup directory of the prod server
backup_path=/mnt/datacamp/backup
# database name on the test server
db=databoard

# remove database on the test server
dropdb -U postgres ${db}
# recreate a db
psql -U postgres -c '\i tools/setup_database.sql'
# Pb with db user password..not properly set with the above script... workaround:
psql -U postgres -c "ALTER ROLE $DATABOARD_DB_USER WITH PASSWORD '$DATABOARD_DB_PASSWORD'"
python manage.py db upgrade
# Restore the db
pg_restore -j 8 -U postgres -d ${db} ${prod_db_dump}
# rsync databoard files
rsync -a $backup_path/databoard ${databoard_path}/

exit 0
