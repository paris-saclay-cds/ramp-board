#!/bin/sh
# delete db backup files which are older than 10 days
path=/mnt/datacamp/backup
logfile=${path}/log_housekeeping
day=10

rm -f $logfile
for file in `find $path -mtime +$day -type f -name *.tar`
do
    echo "deleting: " $file >> $logfile
    rm $file
done

exit 0
