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




### Test ramp locally

fab test_setup
fab serve
 - goto http://0.0.0.0:8080/ and test the interface

### Publish on the server

 - setup ramps_configs['<ramp_name>_remote'] in config.py
 - in development dir (repeat each time you change the code):
fab publish_sofware:target=test/production
fab publish_problem:<problem_name>,target=local/test/production
cd /mnt/datacamp/code
pip install -Ur requirements.txt
python setup.py develop

### Test and setup ramp on remote server

cd /mnt/datacamp/databoard_test or /mnt/datacamp/databoard
fab test_setup
fab serve:<port>
 - open server, log in (with one of the accounts set up in tests/test_model)
 - submit the starting_kit under the name 'test'
fab train_test:<event_name>
 - check the leaderboard, private_leaderboard, user_interactions, etc.

### Set up the database

 - old setup, we keep it for a while in case we migrate old databases.
 - make a file users_to_add.csv with fields
 firstname,lastname,email,name,hidden_notes,access_level
 - if passwords are needed, run this to generate 'passwords.csv'
fab generate_passwords:users_to_add.csv,passwords.csv
 - eventually edit it (if you eg want to use existing passwords)
 - rsync it to the remote server, eg.
rsync -rRultv user_batches/m2_20160121 root@onevm-177.lal.in2p3.fr:/mnt/datacamp/databoard_air_passengers_8080/
 - on the server (users should be the same in the same order, user_name is not checked) (maybe turn config.is_send_trained_mails off)
fab add_users_from_file:users_to_add.csv,passwords.csv
 - log in under other user, check site
 - when everything is fine, send mail (port can overwrite config.config_object.server_port in the mail):
fab send_password_mails:passwords.csv,port=<port> 


## Command description 
    
### Launch the web server

fab serve:<port>

### Other commands
