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
- fab setup
- fab fetch
- fab train
- fab leaderboard 

### Launch the web server

- `export SERV_PORT=<port number>  # 8080 by default`
- `fab serve` 

### Other commands

- fab clear_cache  # clear joblib cache
- fab clear_db  # clear the database
- fab clear_registrants  # clear the teams repositories
- fab all # equivalent to fab fetch train leaderboard 



