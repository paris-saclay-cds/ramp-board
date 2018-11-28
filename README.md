# RAMP bundle

[![Build Status](https://travis-ci.com/paris-saclay-cds/ramp-board.svg?branch=master)](https://travis-ci.com/paris-saclay-cds/ramp-board)

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


This repository contains the following RAMP modules:

- [`ramp-database`](ramp-database/README.md) - RAMP database module
- [`ramp-engine`](ramp-engine/README.md) - RAMP runner service
- [`databoard`](databoard/README.md) - RAMP frontend server

The modules can be installed independantly but have been added to the same repository so they can be kept in sync.


Installation
------------

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


Configuration
-------------

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


Testing
-------

All modules can be tested with [`pytest`][pytest]

```bash
pytest <module>
```

Note that testing requires a database to be [setup and running][dbsetup].


[pytest]: https://docs.pytest.org/en/latest/
[dbsetup]: ramp-database/README.md#set-up-of-a-postgresql-database