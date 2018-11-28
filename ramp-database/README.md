RAMP database model
===================

This package contains the ORM model for the RAMP database.

## Set up of a PostgreSQL database

1. create db directory and initialise the PostgreSQL database
    ```bash
    mkdir postgres_dbs
    initdb postgres_dbs
    ```
2. start the PostgreSQL engine
    ```bash
    pg_ctl -D postgres_dbs -l postgres_dbs/logfile start
    ```
3. create a user `<db_user>` and set a password `<db_password>`
    ```bash
    createuser --pwprompt <db_user>
    ```
4. create the database called `<db_name>` for that user
    ```bash
    createdb --owner=<db_user> <db_name>
    ```

Then set the deployment path and URL address according to the name and path you used, either in the [configuration file][cfgfile], or via [environment variables][envvar].

**WARNING**: make sure you set a different path/url for the test and the production database.

```bash
DATABOARD_DB_URL_TEST=postgresql://<db_user>:<db_password>@localhost/<db_name_test>
DATABOARD_DEPLOYMENT_PATH_TEST=/path/to/<db_name_test>

DATABOARD_DB_URL=postgresql://<db_user>:<db_password>@localhost/<db_name_prod>
DATABOARD_DEPLOYMENT_PATH=/path/to/<db_name_prod>
```

[cfgfile]: ../README.md#json-configuration-file
[envvar]: ../README.md#environment-variables
