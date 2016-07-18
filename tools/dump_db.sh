#!/bin/sh    
# Backup databoard database and databoard files
databoard_path=/mnt/ramp_data/datacamp/databoard
backup_path=/mnt/datacamp/backup
db=databoard

# dump db
date=`date +"%Y%m%d_%H%M%N"`
filename="${backup_path}/${db}_${date}.dump"
pg_dump -U postgres -Fc $db > $filename
# to restore: pg_restore -j 8 -U postgres -d myDB myDB.dump

# rsync databoard files
rsync -a ${databoard_path}/ $backup_path/databoard

exit 0
