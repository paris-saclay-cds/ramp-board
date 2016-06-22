### How to deploy databaord on a Ubuntu 14.04 VM? 

For the production site, use of `deploy_databoard.sh`:   
- deployment with `apache` and `mod_wsgi`  
- use of a postgre database  
- use of the native python and pip     
- databoard code is in `/home/code/databoard`  
- databoard submissions, problems,... are in `/home/datacamp/databoard`  
- backup of `/home/datacamp/databoard` and of the database are made two times per day on the sciencefs disk, which is mounted on `/mnt/datacamp`. Backup are made on `/mnt/datacamp/backup` 


For test purposes, the **main steps** are summed up below (for Ubuntu):  
1. Install apache and mod_wsgi: line 25 of `deploy_databoard.sh`  
2. Install postgres and setup database: lines 46-57 of `deploy_databoard.sh`   
3. Install databoard: lines 61-63 of `deploy_databoard.sh`   
4. Configure Apache: lines 71-91 of `deploy_databoard.sh`.  
5. Set up permissions so that apache can access the app files: lines 85-86 of `deploy_databoard.sh`.   
6. Restart apache: line 91 of `deploy_databoard.sh`.  
