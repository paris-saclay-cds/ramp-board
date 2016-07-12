# Databoard

## Dependencies

Install dependencies with `pip install -Ur requirements.txt`
(You might want to create a virtualenv beforehand)

## Deploy

### Prepare starting kit bundle, data, and test submissions for local test

 - cd problems/<problem_name>
 - starting_kit/<event_name>_ramp_staring_kit.py
 - starting_kit/public_train.csv (+ other data if needed)
 - starting_kit/*.py for the initial submissions in the sandbox
 - starting_kit/user_test_submission.py
 - data/raw/* all the data which prepare_data() will use + the directory structure, usually data/private
 - for local tests (optional), create deposited_submissions/<test_team>/<test_submission> and put the test submission files there
 - run 
zip -r starting_kit.zip starting_kit. 
   This is a bit tricky: if the public_train is created in prepare_data, after deployment (fab add_problem, fab add_event), zip should be re-run just in case.
 - run
fab publish_problem:<problem_name>,target=local/test/production
   which will copy everything on the local test, production or test server

### Add workflow element types and workflows (in case the ramp needs a new one)

 - code the workflow element type in databoard/specific/workflows
 - add it to fabfile.software
 - workflow element type example
fab add_workflow_element_type:classifier,code
 - workflow example:
fab add_workflow:feature_extractor_classifier_calibrator_workflow,feature_extractor,classifier,calibrator

### Add score types (in case the ramp needs a new one)

 - code the score type in databoard/specific/score_types
 - add it to fabfile.software
 - score type example
fab add_score_type:test_score_type,True,0.0,inf

### Add problem
 
 - code the problem in databoard/specific/problems
 - add it to fabfile.software
 - problem example
fab add_problem:variable_stars

### Add event
 
 - code the event in databoard/specific/events
 - add it to fabfile.software
 - event example
fab add_event:variable_stars

## Backup

rsync -rultv root@134.158.75.241:/mnt/datacamp/databoard/db/databoard.db db/
rsync -rultv root@134.158.75.241:/mnt/datacamp/databoard/submissions ./

### Test ramp locally

fab test_setup
fab serve
 - goto http://0.0.0.0:8080/ and test the interface

### Publish on the server

fab publish_software:target=production
fab publish_software:target=test

### If code is redeployed

pip install -Ur requirements.txt
python setup.py develop

### Server

 - old:
fab serve:80 > server_logs/server16.txt 2>&1
 - new:
sudo service apache2 restart
tail -f /var/log/apache2/error.log
sed -i "s#os.environ.get('DATABOARD_DB_URL')#'$DATABOARD_DB_URL'#g" /home/datacamp/code/databoard/config.py

### App performance

#### Profiling
fab profile:port,profiling_output_file  
By default `port=None` (for local profiling) and `profiling_output_file=profiler.log`  
#### Database performance
To report in the logging system queries that takes too long, define an environment variable `DATABOARD_DB_PERF` (equals to 'True' for instance).   
#### Stress Tests with [Locust](http://locust.io/)  
1. Modify `locustfile.py` to specify the current ramp url to be tested and the databoard path (or to add some tasks)
2. Define two environment variables `DATABOARD_USERNAME` and `DATABOARD_PASSWORD` to login during tests.     
3. Set `WTF_CSRF_ENABLED` to `False` in `databoard/config.py`
4. Run `locust -f locustfile.py` 
5. Go to http://127.0.0.1:8089/ and enter the number of users to simulate  


### Example sequence of adding the drug_spectra ramp

 - drug_spectra
fab add_score_type:error_mare_mixed,True,0.0,inf
fab add_score_type:error_mixed,True,0.0,1.0 
fab add_score_type:mare_mixed,True,0.0,inf 
fab add_score_type:mare,True,0.0,inf
fab add_workflow_element_type:feature_extractor_clf,code
fab add_workflow_element_type:feature_extractor_reg,code 
fab add_workflow:feature_extractor_classifier_regressor_workflow,feature_extractor_clf,classifier,feature_extractor_reg,regressor
fab add_problem:drug_spectra 
fab add_event:drug_spectra

## Remount disk

export SCIENCEFS_LOGIN='balazs.kegl'
sshfs -o Ciphers=arcfour256 -o allow_other -o IdentityFile=/root/.ssh/id_rsa_sciencefs -o StrictHostKeyChecking=no "$SCIENCEFS_LOGIN"@sciencefs.di.u-psud.fr:/sciencefs/homes/"$SCIENCEFS_LOGIN"/databoard /mnt/datacamp



---------------
1. Production server

- Production server deployed on prod_ramp 134.158.75.211. 
This instance built from image "test_ramp_060716". This image has been obtained by running the script tools/deploy_databoard.sh.

- Two disks are mounted to the VM:
    * sciencefs disk, which is used for backup. It is mounted to /mnt/datacamp.
    * persistent volume (prod_ramp), where databoard code and submission files are stored. It is mounted to /mnt/ramp_data.

- Databoard code is in /mnt/ramp_data/code/databoard

- Databoard submission files, fabfile.py, ... are in /mnt/ramp_data/datacamp/databoard

- Backup are made every day around midnight and are saved (on the sciencefs disk) in /mnt/datacamp/backup. It is made using cron and bash scripts in /mnt/ramp_data/code/databoard/tools/dump_db.sh + housekeeping.sh.
During RAMP, it might be better to increase the backup frequency. To do this, the crontab file can be edited by running "crontab -e".
Two types of backup:
    * dump of the database
    * rsync of /mnt/ramp_data/datacamp/databoard folder

2. Test server

- Test server deployed on test_ramp 134.158.75.119

- One disk is mounted to the VM:
    * persistent volume (test_ramp), where databoard code and submission files are stored. It is mounted to /mnt/ramp_data.

- Databoard submission files, fabfile.py, ... are in /mnt/ramp_data/datacamp/databoard

- Databoard code is in /mnt/ramp_data/code/databoard

- If you want to use databoard submission files and database of the production server, you can use the script tools/prod_db_to_test.sh from the test server. Be careful to change the name of the database dump you want to use.
