#!/bin/bash 
# :Usage: bash script_install/master_workers.sh {start|stop|restart} {nb_local_workers}
# Starting/stopping/restarting nb_local_workers locally for the scheduler 

mkdir celery_info

export LOCAL_WORKERS=""
for i in `seq 1 $2`;
do
    export LOCAL_WORKERS=("$LOCAL_WORKERS lw$i"); 
done    

if [ $1 = "stop" ]; then
    echo "Stopping the workers";
    # Local workers
    celery multi $1 $LOCAL_WORKERS --pidfile="$(pwd)/celery_info/%n.pid";
else
    echo "$1 the workers";
    # Local workers and starting the scheduler
    celery multi $1 $LOCAL_WORKERS -l INFO -A databoard \
        --logfile="$(pwd)/celery_info/%n.log" \
        --pidfile="$(pwd)/celery_info/%n.pid";
    celery -A datarun beat -s "$(pwd)/celery_info/celerybeat-schedule" \
        --pidfile="$(pwd)/celery_info/celerybeat.pid" --detach

fi
