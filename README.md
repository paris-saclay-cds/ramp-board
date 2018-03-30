# RAMP backend

[![Build Status](https://travis-ci.org/paris-saclay-cds/ramp-backend.svg?branch=master)](https://travis-ci.org/paris-saclay-cds/ramp-backend)

Suite of command-line tools to interact with the RAMP database from an external backend server.


## Command-line utilities

#### `ramp_new_submissions`

```bash
ramp_new_submissions <backend_config> --event_name=<event>
```

#### `ramp_set_state`

```bash
ramp_set_state <backend_config> -E <event> -T <team> <submission_name> <new state>
```
