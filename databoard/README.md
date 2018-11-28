Databoard
=========


Installation
------------

1. set up the RAMP database (see [here][setdb])

2. set up the deployment path (see [config][envvar])

3. create the following directories
    ```bash
    mkdir -p $DATABOARD_DEPLOYMENT_PATH_TEST
    mkdir $DATABOARD_DEPLOYMENT_PATH_TEST/ramp-kits 
    mkdir $DATABOARD_DEPLOYMENT_PATH_TEST/ramp-data
    mkdir $DATABOARD_DEPLOYMENT_PATH_TEST/submissions
    ```

[setdb]: ../ramp-database/README.md#set-up-of-a-postgresql-database
[envvar]: ../README.md#environment-variables


### Adding a new problems

```
cd <frontend>/ramp-kits
git clone https://github.com/ramp-kits/<problem>
cd <problem>
jupyter nbconvert --to html <problem>_starting_kit.ipynb
```

### Get the prod database  

A dump of the prod database is saved everyday on the scienceFS backup disk. You can use this dump to populate your test db. You need access to the scienceFS backup disk and the prod database  credentials.  


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
zip -r starting_kit.zip starting_kit
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

### Server

 - old:
fab serve:80 > server_logs/server16.txt 2>&1
 - new:
sudo service apache2 restart

 - inspect server log file:
tail -n1000 -f /var/log/apache2/error.log

sed -i "s#os.environ.get('DATABOARD_DB_URL')#'$DATABOARD_DB_URL'#g" /home/datacamp/code/databoard/config.py


### Example sequence of adding ramps

#### Titanic

```
fab add_score_type:auc,"0",0.0,1.0
fab add_workflow:feature_extractor_classifier_workflow,feature_extractor,classifier
fab add_problem:titanic
fab add_event:titanic
fab sign_up_team:titanic,kegl
```

#### Pollenating insects 2

```
fab add_score_type:f1_above,"0",0.0,1.0
fab add_problem:pollenating_insects_2,force=True
fab add_event:pollenating_insects_2_paillasse,force=True
fab sign_up_team:pollenating_insects_2_paillasse,kegl
fab sign_up_team:pollenating_insects_2_paillasse,mcherti
```

#### Pollenating insects

```
fab add_workflow_element_type:image_preprocessor,code
fab add_workflow_element_type:batch_classifier,code
fab add_workflow:batch_classifier_workflow,image_preprocessor,batch_classifier
fab add_problem:pollenating_insects
fab add_event:pollenating_insects_M1XMAP583_M2HECXMAP542_201617
fab sign_up_team:pollenating_insects_M1XMAP583_M2HECXMAP542_201617,kegl
fab sign_up_team:pollenating_insects_M1XMAP583_M2HECXMAP542_201617,mcherti
```

#### HEP tracking

```
fab add_score_type:clustering_efficiency,"0",0.0,1
fab add_workflow_element_type:clusterer,code
fab add_workflow:clusterer_workflow,clusterer
fab add_problem:HEP_tracking
fab add_event:HEP_tracking
fab sign_up_team:HEP_tracking,kegl
```

#### drug_spectra

```
fab add_score_type:error_mare_mixed,"1",0.0,inf
fab add_score_type:error_mixed,"1",0.0,1.0
fab add_score_type:mare_mixed,"1",0.0,inf
fab add_score_type:mare,"1",0.0,inf
fab add_workflow_element_type:feature_extractor_reg,code
fab add_workflow_element_type:feature_extractor_clf,code
fab add_workflow:feature_extractor_classifier_regressor_workflow,feature_extractor_clf,classifier,feature_extractor_reg,regressor
fab add_problem:drug_spectra
fab add_event:drug_spectra
fab sign_up_team:drug_spectra,kegl
```

#### air passengers

```
fab add_workflow_element_type:external_data,data
fab add_workflow:feature_extractor_regressor_with_external_data_workflow,feature_extractor,regressor,external_data
fab add_problem:air_passengers
fab add_event:air_passengers_dssp4
fab sign_up_team:air_passengers_dssp4,kegl
fab sign_up_team:air_passengers_dssp4,agramfort
```

#### sea ice

```
fab add_workflow_element_type:ts_feature_extractor,code
fab add_workflow:ts_feature_extractor_regressor_workflow,ts_feature_extractor,regressor
fab add_problem:sea_ice
fab add_event:sea_ice_colorado
fab sign_up_team:sea_ice_colorado,kegl
```

#### el nino (the first two lines are unnecessary if sea ice is already there)

```
fab add_workflow_element_type:ts_feature_extractor,code
fab add_workflow:ts_feature_extractor_regressor_workflow,ts_feature_extractor,regressor
fab add_problem:el_nino
fab add_event:el_nino
fab sign_up_team:el_nino,kegl
```

#### epidemium2 cancer mortality

```
fab add_workflow_element_type:feature_extractor,code
fab add_workflow_element_type:regressor,code
fab add_workflow:feature_extractor_regressor_workflow,feature_extractor,regressor
fab add_problem:epidemium2_cancer_mortality
fab add_event:epidemium2_cancer_mortality
fab sign_up_team:epidemium2_cancer_mortality,kegl
```

### Batch sign up users
 - make a file users_to_add.csv with header
 firstname  lastname  email name  hidden_notes
 - make passwords for them:
fab generate_passwords:users_to_add.csv,users_to_add.csv.w_pwd
 - add them to the ramp (it's a bit messy now when a user is already there with a mail but different username; should be handled)
fab add_users_from_file:users_to_add.csv,users_to_add.csv.w_pwd
 - sign them up to an event:
fab sign_up_event_users_from_file:users_to_add.csv,<event>
 - send them mails with their passwords:
fab send_password_mails:users_to_add.csv.w_pwd


## DATABASE MIGRATION

Then you can setup or upgrade the database with:

    python manage.py db upgrade

Run: `python manage.py db migrate`. It creates a migration file in `migrations/versions/`
Add `import databoard` on top of the migration file
Run: `python manage.py db upgrade` to apply the migration
**Don't forget to add and commit migrations files**


## On the backend

export OMP_NUM_THREADS=1


## Backup of db

Define the following environment variables:  

    export DATABOARD_DB_NAME='databoard'
    export DATABOARD_DB_USER='<prod_db_user>'     # Ask Balazs
    export DATABOARD_DB_PASSWORD='<prod_db_pwd>'  # Ask Balazs
    export SCIENCEFS_LOGIN='balazs.kegl'          # You need the private key or the password...
    export SCIENCEFS_ID='<scienceFS_key>'         # Path and name of the scienceFS private key
    export mount_path='mount_backup'              # Path where to mount the scienceFS disk to get backups
    export DATABOARD_PATH='/tmp'                  # Root path to your databoard app. By default, on your local computer it is /tmp, so that the app is in /tmp/datacamp/databoard. For prod and test servers it is '/mnt/ramp_data'.                     
    export prod_db_dump='<blabla.dump>'           # db dump to be used (just write the dump name without the path to it

Run script: ``bash tools/prod_db_to_test.sh``

