# Databoard

## Dependencies

Install dependencies with `pip install -Ur requirements.txt`
(You might want to create a virtualenv beforehand)

## Deploy

### Create specific.py

 - put it into databoard/ramps/<ramp_name>
 - along with __init__.py
 - add databoard.ramps.<ramp_name> to packages in setup.py

### Setup data

 - create ramps/<ramp_name>/data /public /private /raw
 - put the data file into amps/<ramp_name>/data/raw

### Setup the sandbox

 - create ramps/<ramp_name>/sandbox and put their the starting kit submission

### Setup the test submissions

 - for local tests, create ramps/<ramp_name>/deposited_submissions/<test_team>/<test_submission> and put the test submission files there

### Test ramp locally

 - put in config.py
ramps_configs['<ramp_name>_local'] = RampConfig(
    ramp_name='<ramp_name>', **local_kwargs)
 - in development dir:
fab publish_test:<ramp_name>_local
cd /tmp/databoard_<ramp_name>_8080
fab test_ramp
fab serve
 - goto http://0.0.0.0:8080/ and test the interface

### Publish on the server

 - setup ramps_configs['<ramp_name>_remote'] in config.py
 - in development dir (repeat each time you change the code):
fab publish:<ramp_name>_remote
 - publish data:
fab publish_data:<ramp_name>_remote
 - on server:
 - if first time code is deployed, or setup.py changed:
cd /mnt/datacamp/code
pip install -Ur requirements.txt
python setup.py develop

### Test and setup ramp on remote server

cd /mnt/datacamp/databoard_<ramp_name>_<port>
fab test_ramp
 - this will also set the ramp up (but not the database)
fab serve:<port>
 - open server, log in (with one of the accounts set up in tests/test_model)
 - submit the sandbox under the name 'test'
fab train_test
 - check the leaderboard, private_leaderboard, user_interactions, etc.

### Set up the database

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
