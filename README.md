
# Databoard

## Dependencies

Install dependencies with `pip install -Ur requirements.txt`
(You might want to create a virtualenv beforehand)

## Setting up the data set
The data must be input inside the `input/` folder.


## Deploy

- Open fabfile.py and modify the server address and the deployment folder.
- `fab publish  # does rsync`
- `ssh <server addr>`
- `pip install -Ur requirements.txt`
- `screen -S databoard`
- `export SERV_PORT=<port number>  # 8080 by default`
- `fab serve` 
- Detach from screen (ctrl+a, ctrl+d)
- `fab setup`
- `fab fetch train leaderbaord`


## Command description 
- `fab setup`
	- to reinitialize the registrations and the joblib cache, use `fab setup:wipeall:1`
- `fab fetch`
- `fab train[:lb=<leaderboard parameters>][state=<specific state to train>]`  # use 'all' to force train all the models
- `fab leaderboard[:which=<leaderboards>]`
	- `which` can be `classical`, `combined`, `times`, `test`, or `all` (default).
    - `all` doesn't contain test
    
### Launch the web server

- `export SERV_PORT=<port number>  # 8080 by default`
- `fab serve` 

### Other commands

- `fab clear_cache  # clear joblib cache`
- `fab clear_db  # clear the database`
- `fab clear_registrants  # clear the teams repositories`
- `fab all # equivalent to fab fetch train leaderboard`
- `fab print_db[:table=<table name>,state=<model state>]`
	-  e.g. `fab print_db:table=models,state=error`
	- or simply `fab print_db:models,trained`



