Set up your own server to run RAMP
==================================

The first step to setup your server is to install the RAMP bundle of packages.
Refer to :ref:`install guideline <install>`.

Once the packages install, proceed to the following steps.

Set up the RAMP database
------------------------

We advise to use ``PostgreSQL`` for managing the database. You can install
``PostgreSQL`` using the system packages. If you are using ``conda``, we
advise you to install it using the following command::

    conda install postgresql

FIXME: I think that we can automatized this part.
Once ``PostgreSQL`` install, we need to create a directory where to store the
Postgres databases::

    mkdir postgres_dbs

Then, initialize the Postgre databases::

    initdb prosgres_db

Once the databases have been initialized, we need to create an admin user and
create the database which will be used for RAMP::

    createuser --pwprompt <db_user>
    createdb --owner=<db_user> <db_name>

You will need to use <db_user>, <user_password>, and <db_name>. We will use
these information in the RAMP configuration files.

Set up the RAMP deployment for kits, data, and submissions
----------------------------------------------------------

You need to create a directory in which you will deploy the RAMP kits, data,
and submissions::

    mkdir ramp_deployment
    cd ramp_deployment

FIXME: we should not copy the configuration file. You need to copy the
``database_config.yml`` file located in ``ramp-utils/ramp_utils/template`` and
rename it ``config.yml``.

This config file should look like::

    flask:
        secret_key: abcdefghijkl
        wtf_csrf_enabled: true
        log_filename: None
        max_content_length: 1073741824
        debug: true
        testing: false
        mail_server: smtp.gmail.com
        mail_port: 587
        mail_default_sender: ['RAMP admin', 'rampmailer@gmail.com']
        mail_username: user
        mail_password: password
        mail_recipients: []
        mail_use_tls: false
        mail_use_ssl: true
        mail_debug: false
        sqlalchemy_track_modifications: true
    sqlalchemy:
        drivername: postgresql
        username: <db_user>
        password: <user_password>
        host: localhost
        port: 5432
        database: <db_name>

You will need to change the information regarding the database and the mail
information.

Create an admin user
--------------------

To operate the event, it is useful to first create an admin user::

    ramp-database add-user --login admin_user --password password --firstname firstname --lastname lastname --email admin@email.com --access-level admin

Launching the RAMP website
--------------------------

FIXME: we need to update this stage
At this stage, you will be able to launch the RAMP website::

    ramp-frontend launch
