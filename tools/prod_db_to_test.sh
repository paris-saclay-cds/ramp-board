#!/bin/sh    
# Copy databoard files and database from production sever to test server (or locally).
# This script has to be run from the test server in the databoard code directory 
# The script assumes that:
# - sciencefs disk (used to backup prod server) is mounted 
# - migration files are up-to-date on test server
# on the test server (in backup_path)
# The current test database is removed and replace by a copy of the prod database

# db dump to be used to recreate the test database
prod_db_dump=databoard_blabla.dump
# backup directory of the prod server
backup_path=/mnt/datacamp/backup
# databoard files directory on the test server
databoard_path=/mnt/ramp_data/datacamp/databoard
# path where to mount the sciencefs disk on the test server
mount_path = /mnt/datarun
# database name on the test server
db=databoard

# mount sciencefs disk
sshfs -o Ciphers=arcfour256 -o allow_other -o IdentityFile=/root/.ssh/id_rsa_sciencefs -o StrictHostKeyChecking=no "$SCIENCEFS_LOGIN"@sciencefs.di.u-psud.fr:/sciencefs/homes/"$SCIENCEFS_LOGIN"/databoard ${mount_path}

# remove database on the test server
dropdb -U postgres ${db}
# recreate a db
psql -U postgres -c '\i tools/setup_database.sql'
# Pb with db user password..not properly set with the above script... workaround:
psql -U postgres -c "ALTER ROLE $DATABOARD_DB_USER WITH PASSWORD '$DATABOARD_DB_PASSWORD'"
# Restore the db
pg_restore -j 8 -U postgres -d ${db} ${prod_db_dump}
# rsync databoard files
rsync -a $backup_path/databoard ${databoard_path}/

# unmount sciencefs disk
fusermount -u ${mount_path}
exit 0
