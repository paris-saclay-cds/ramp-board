RAMP database model
===================

This package contains the ORM model for the RAMP database.


## Set up of a PostgreSQL database

```bash
# initialise the PostgreSQL database 
mkdir postgres_dbs && initdb postgres_dbs
pg_ctl -D postgres_dbs -l postgres_dbs/logfile start
# create a user and set a password
createuser --pwprompt <username>
# create the database
createdb --owner=<username> <db_name>
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