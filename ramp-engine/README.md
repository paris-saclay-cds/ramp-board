# RAMP backend

[![Build Status](https://travis-ci.org/paris-saclay-cds/ramp-backend.svg?branch=master)](https://travis-ci.org/paris-saclay-cds/ramp-backend)

Suite of command-line tools to interact with the RAMP database from an external backend server.


## Command-line utilities

#### `ramp_new_submissions`

```bash
ramp_new_submissions <backend_config> <event_name>
```

#### `ramp_set_state`

```bash
ramp_set_state <backend_config> <submission_id> <new_state>
```


## API

To use the RAMP backend tools in python scripts, the same functionalities
are provided through an API.

Here is an example of how to use it.

```python
from rampbkd import get_submissions
from rampbkd import set_submission_state

# Get a list of new submissions from a given event
new_submissions = get_submissions('backend_config.yml',
                                  event_name='mars_craters',
                                  state='new')

sub_id0, sub_files = new_submissions[0]

# Download submission files and send it to training.

# Then change the submission status on the RAMP database
set_submission_state('backend_config.yml',
                     submission_id=sub_id0,
                     state='send_to_training')
```
