#!/bin/sh    
# Backup databoard database and databoard files
databoard_path=/home/datacamp/databoard
backup_path=/mnt/datacamp/backup
db=databoard

# dump db
date=`date +"%Y%m%d_%H%M%N"`
filename="${backup_path}/${db}_${date}.tar"
pg_dump -U postgres -Ft $db > $filename

# rsync databoard files
rsync -a ${databoard_path}/ $backup_path/databoard

exit 0
