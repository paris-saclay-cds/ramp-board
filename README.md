# Databoard

## Dependencies

Install dependencies with `pip install -Ur requirements.txt`
(You might want to create a virtualenv beforehand)

## Set up the database

Postgres databases: one for test and one for dev.   
1. Install postgres and create two databases (`createdb db_name`)  
2. Set up environment variables:  
* `DATABOARD_DB_URL`: SQLALCHEMY_DATABASE_URI for the dev database, e.g. `postgresql://localhost/db_name`  
* `DATABOARD_DB_URL_TEST`: SQLALCHEMY_DATABASE_URI for the test database  
* `DATABOARD_TEST`: `True` if you want to use the test database, `False` else.  
3. Upgrade the dev database with: `python manage.py db upgrade`  

### Migrations
Run: `python manage.py db migrate`. It creates a migration file in `migrations/versions/`  
Add `import databoard` on top of the migration file  
Run: `python manage.py db upgrade` to apply the migration
**Don't forget to add and commit migrations files**

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

set the environment variable `DATABOARD_TEST` to `True` (`export DATABOARD_TEST=True`)
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

### Example sequence of adding the drug_spectra ramp

 - drug_spectra
fab add_score_type:error_mare_mixed,True,0.0,inf
fab add_score_type:error_mixed,True,0.0,1.0 
fab add_score_type:mare_mixed,True,0.0,inf 
fab add_score_type:mare,True,0.0,inf
fab add_workflow_element_type:feature_extractor_reg,code 
fab add_workflow_element_type:feature_extractor_clf,code
fab add_workflow:feature_extractor_classifier_regressor_workflow,feature_extractor_clf,classifier,feature_extractor_reg,regressor
fab add_problem:drug_spectra 
fab add_event:drug_spectra

 - air passengers
fab add_workflow_element_type:external_data,data
fab add_workflow:feature_extractor_regressor_with_external_data_workflow,feature_extractor,regressor,external_data
fab add_problem:air_passengers
fab add_event:air_passengers_dssp4
fab sign_up_team:air_passengers_dssp4,kegl
fab sign_up_team:air_passengers_dssp4,agramfort

