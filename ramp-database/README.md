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
createdb --owner=<username> databoard_test
```