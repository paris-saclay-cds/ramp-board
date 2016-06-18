This README explains how to deploy the site with `apache` and `mod_wsgi`. The script `deploy_databoard.sh` does this job on an Ubuntu 14.04 VM for the production site.   
For test purposes, the **main steps** are summed up below (for Ubuntu):  

1. Install apache and mod_wsgi: line 25 of `deploy_databoard.sh`
2. Install postgres and setup database: lines 46-57 of `deploy_databoard.sh`   
3. Install databoard: lines 61-63 of `deploy_databoard.sh`  
4. Configure Apache: lines 71-91 of `deploy_databoard.sh`. 
5. Set up permissions so that apache can access the app files: lines 85-86 of `deploy_databoard.sh`.   
6. Restart apache: line 91 of `deploy_databoard.sh`.  
