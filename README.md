# Databoard

## Dependencies

Install dependencies with `pip install -Ur requirements.txt`
(You might want to create a virtualenv beforehand)

## Configuration

**Copy `databoard/config_local.py` to `databoard/config.py`**. If you need special settings, you can modify `databoard/config.py`.
**Do not commit `databoard/config.py`**, since it might contain passwords...

## Set up the database

To run the test you will need to set environment variable DATABOARD_TEST to True:

    export DATABOARD_TEST=True

otherwise set it to False:

    export DATABOARD_TEST=False

You can use different database system:

**local sqlite**

For a local install (on a unix system) you can do:

    export DATABOARD_DB_URL_TEST=sqlite:////tmp/databoard_test.db
    export DATABOARD_DB_URL=sqlite:////tmp/databoard_test.db

**Postgres databases**: one for test and one for dev.

1. Install postgres and create two databases (`createdb <db_name>`)

For example you do in the postgres terminal: `createdb databoard`

2. Set up environment variables:

    - `DATABOARD_DB_URL`: `SQLALCHEMY_DATABASE_URI` for the dev database, which should be something like `postgresql://<db_user>:<db_password>@localhost/<db_name>`
    - `DATABOARD_DB_URL_TEST`: `SQLALCHEMY_DATABASE_URI` for the test database

Example:

    export DATABOARD_DB_URL=postgresql://$USER:<db_password>@localhost/databoard

Assuming you created a database with `createdb databoard`.

Then you can setup or upgrade the database with:

    `python manage.py db upgrade`

### Migrations

Run: `python manage.py db migrate`. It creates a migration file in `migrations/versions/`
Add `import databoard` on top of the migration file
Run: `python manage.py db upgrade` to apply the migration
**Don't forget to add and commit migrations files**


## Set up datarun  

If you want to use datarun (e.g. for local tests), you need to define the 3 environment variables (with your datarun credentials):  

    export DATARUN_URL='uuuu'
    export DATARUN_USERNAME='vvvv'
    export DATARUN_PASSWORD='wwww'

### How to use datarun to train test submissions?

See datarun documentation (especially "notes for databoard users"):
- [pdf here](https://github.com/camillemarini/datarun/blob/master/docs/datarun.pdf)
- [html here](https://github.com/camillemarini/datarun/tree/master/docs/html)


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

### If code is redeployed

pip install -Ur requirements.txt
python setup.py develop

### Server

 - old:
fab serve:80 > server_logs/server16.txt 2>&1
 - new:
sudo service apache2 restart

sed -i "s#os.environ.get('DATABOARD_DB_URL')#'$DATABOARD_DB_URL'#g" /home/datacamp/code/databoard/config.py

### Mac bug

Add this to ~/.bash_profile:
export DYLD_FALLBACK_LIBRARY_PATH=$HOME/anaconda/lib/:$DYLD_FALLBACK_LIBRARY_PATH

### Example sequence of adding ramps

 - drug_spectra
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

 - air passengers
fab add_workflow_element_type:external_data,data
fab add_workflow:feature_extractor_regressor_with_external_data_workflow,feature_extractor,regressor,external_data
fab add_problem:air_passengers
fab add_event:air_passengers_dssp4
fab sign_up_team:air_passengers_dssp4,kegl
fab sign_up_team:air_passengers_dssp4,agramfort

 - sea ice
fab add_workflow_element_type:ts_feature_extractor,code
fab add_workflow:ts_feature_extractor_regressor_workflow,ts_feature_extractor,regressor
fab add_problem:sea_ice
fab add_event:sea_ice_colorado
fab sign_up_team:sea_ice_colorado,kegl

 - el nino (the first two lines are unnecessary if sea ice is already there)
fab add_workflow_element_type:ts_feature_extractor,code
fab add_workflow:ts_feature_extractor_regressor_workflow,ts_feature_extractor,regressor
fab add_problem:el_nino
fab add_event:el_nino
fab sign_up_team:el_nino,kegl

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

### App performance

#### Profiling
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




### Current deployment on stratuslab (openstack)

#### 1. Production server

- Production server deployed on prod_ramp 134.158.75.211.

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

#### 2. Test server

- Test server deployed on test_ramp 134.158.75.185

- One disk is mounted to the VM:
    * persistent volume (test_ramp), where databoard code and submission files are stored. It is mounted to /mnt/ramp_data.

- Databoard submission files, fabfile.py, ... are in /mnt/ramp_data/datacamp/databoard

- Databoard code is in /mnt/ramp_data/code/databoard

- If you want to use databoard submission files and database of the production server, you can use the script tools/prod_db_to_test.sh from the test server. Be careful to change the name of the database dump you want to use.


#### Remount disks on prod server

##### sciencefs disk

export SCIENCEFS_LOGIN='balazs.kegl'

sshfs -o Ciphers=arcfour256 -o allow_other -o IdentityFile=/root/.ssh/id_rsa_sciencefs -o StrictHostKeyChecking=no "$SCIENCEFS_LOGIN"@sciencefs.di.u-psud.fr:/sciencefs/homes/"$SCIENCEFS_LOGIN"/databoard /mnt/datacamp

##### prod_ramp disk

mount dev_file /mnt/ramp_data

where dev_file corresponds to the path of the dev file
of the prod_ramp disk : <https://keystone.lal.in2p3.fr/dashboard/project/volumes/>.

Currently it is : /dev/vdb, so the command is :

mount /dev/vdb /mnt/ramp_data


### How to deploy databoard on stratuslab

A databoard server needs:
- an **Ubuntu 14.04 VM** with databoard installed on it
- a **persistent disk** where submission files and data are saved
- the **sciencefs disk** where are saved **backups** of the postgres database and of submission files (**only for a production server**)

Below are the instructions to **start a databoard server using the latest state of the production database**:

#### For a test server

1. Start a VM (Ubuntu 14.04) on openstack (via the openstack interface).
2. Go to `databoard/tools directory` and Make it possible to log in as root: `ssh ubuntu@<VM_IP_ADDRESS> 'bash -s' < root_permissions.sh`
3. Create a persistent disk and attach it to the VM (via the openstack interface).
4. Create a file `env.sh` which contain required environment variables **DO NOT COMMIT THIS FILE**:
```
export DATABOARD_PATH='/mnt/ramp_data/'  #where to mount the persistent disk
export DATABOARD_DB_NAME='databoard'
export DATABOARD_DB_USER='xxxx'
export DATABOARD_DB_PASSWORD='yyyy'
export DATABOARD_DB_URL='postgresql://xxxx:yyyy@localhost/databoard'
export SCIENCEFS_LOGIN='zzzz'
export DATARUN_URL='uuuu'
export DATARUN_USERNAME='vvvv'
export DATARUN_PASSWORD='wwww'
```
5. scp to the VM the file `env.sh` and the script `deploy_databoard.sh`: `scp env.sh deploy_databoard.sh root@<VM_IP_ADDRESS>:/root/.`
6. ssh to the instance and run `bash deploy_databoard.sh {disk_path} {db_dump}` where `disk path` is the path to the attached disk (something like `/dev/vdb`, which can be found on the openstack interface) and database dump from which to create new database (give only the dump file name, this file should be located on the sciencefs disk in `~/databoard/backup` which will be mounted on the VM in `/mnt/datacamp/backup`). This script:
    - installs databoard on the VM. It will clone the project from git (line 112). **Modify this line to clone it with your account if needed**.
    - mounts the sciencefs disk to retrieve backups of the latest state of the db and of associated submission files
    - mounts the persistent disk and copy onto it backups of submission files from the sciencefs disk
    - installs apache, ... and start the application
    - starts celery workers to send jobs to datarun  6. Unmount the sciencefs disk `fusermount -u /mnt/datacamp`

#### For a production server

Follow **instructions 1 to 6** from above.
7. Set up backups of the db and of submission files: use of `crontab` to run `tools/dump_db.sh` and `tools/housekeeping.sh`. To set up it, add these lines to the file opened by running `crontab -e`:
```
02 0    * * *   root    bash /mnt/ramp_data/code/databoard/tools/dump_db.sh
22 1    * * *   root    bash /mnt/ramp_data/code/databoard/tools/housekeeping.sh
```
