RAMP database model
===================

This package contains the ORM model for the RAMP database.

Set up of a PostgreSQL database
-------------------------------

1. create db directory and initialise the PostgreSQL database::

    mkdir postgres_dbs
    initdb postgres_dbs

2. start the PostgreSQL engine::

    pg_ctl -D postgres_dbs -l postgres_dbs/logfile start

3. create a user `<db_user>` and set a password `<db_password>`::

    createuser --pwprompt <db_user>

4. create the database called `<db_name>` for that user::

    createdb --owner=<db_user> <db_name>
