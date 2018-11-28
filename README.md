# RAMP bundle

[![Build Status](https://travis-ci.com/paris-saclay-cds/ramp-board.svg?branch=master)](https://travis-ci.com/paris-saclay-cds/ramp-board)

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


This repository contains the following RAMP modules:

- [`ramp-database`](ramp-database/README.md) - RAMP database module
- [`ramp-engine`](ramp-engine/README.md) - RAMP runner service
- [`databoard`](databoard/README.md) - RAMP frontend server

The modules can be installed independantly but have been added to the same repository so they can be kept in sync.

## Installation

1. Retrieve the main repository

    ```bash
    git clone https://github.com/paris-saclay-cds/ramp-board
    cd ramp-board
    ```

2. Install Python dependencies using `conda` or `pip`

    - with `conda`

      ```bash
      # Make sure you run the latest version of conda
      conda update conda
      # Set up the virtual environment
      conda env create -f environment.yml
      # Activate it
      conda activate ramp-server
      ```

    - with `pip`

      ```bash
      pip install -r requirements.txt
      ```

3. Install all the elements at once
    
    ```bash
    make install
    ```
  
   or each project independently 

    ```bash
    cd ramp-<project>
    pip install .
    ```

## Configuration

`databoard` comes with a default configuration that can be overwritten via environment variables or a JSON configuration file.

**Note:** If you use both (e.g. general JSON config file AND environment variables for sensitive parameters like passwords), know that the user configuration file is **loaded last** so it will take precedence over the environment variables. You should therefore remove these entries from the JSON file.

### JSON configuration file


The `DATABOARD_USER_CONFIG` environment variable must be set to the absolute path of the config file in order to be used by the `ramp-server`.

```bash
export DATABOARD_USER_CONFIG=/path/to/userconfig.json
```

### Environment variables

```bash
export DATABOARD_STAGE=TESTING (or PRODUCTION)
```

## Initial set up of the database

```bash
# initialise the PostgreSQL database 
mkdir postgres_dbs && initdb postgres_dbs
pg_ctl -D postgres_dbs -l postgres_dbs/logfile start
# create a user and set a password
createuser --pwprompt <username>
# create the database
createdb --owner=<username> databoard_test
```

`<username>` should be the user specified in `$DATABOARD_DB_USER`. 
It will prompt you for password which should be the same specified in `$DATABOARD_DB_PASSWORD`.

## Running tests

```
git clone https://github.com/paris-saclay-cds/ramp-board.git
cd ramp-board
pip install .
make tests
```
or
```
make test-all
```
The deployment directory is `/tmp/databoard`. You can change it in `config.py`
```
cd /tmp/databoard
fab serve
```
