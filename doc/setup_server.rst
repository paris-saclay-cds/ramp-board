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
Once ``PostgreSQL`` is installed, we initialize a Postgres database cluster::

    initdb postgres_dbs

Once the cluster has been initialized, we need to create an admin user and
create the database which will be used for RAMP::

    createuser --pwprompt <db_user>
    createdb --owner=<db_user> <db_name>

where <db_user>, <db_password> and <db_name> should be replaced with actual
values. Those values should match with the RAMP configuration file.

Set up the RAMP deployment for kits, data, and submissions
----------------------------------------------------------

You need to create a directory in which you will deploy the RAMP kits, data,
and submissions::

    mkdir ramp_deployment
    cd ramp_deployment

Next, you need to create a ``config.yml`` file in the directory, which holds
the configuration for the database and the flask server. A generic template
can be created using::

    ramp init

which will create a ``config.yml`` file inside the deployment directory. Now,
you can edit this file filling in the correct database name, user name and
password, and filling in a flask secret key.

This config file should look like::

    flask:
        secret_key: abcdefghijkl
        mail_server: smtp.gmail.com
        mail_port: 587
        mail_default_sender: ['RAMP admin', 'rampmailer@gmail.com']
        mail_username: user
        mail_password: password
        mail_recipients: []
        mail_use_tls: false
        mail_use_ssl: true
        mail_debug: false
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

Launching a test instance of the  RAMP website
----------------------------------------------

At this stage, you will be able to launch the RAMP website::

    ramp frontend test-launch

This uses the built-in server of Flask suitable for testing. To deploy it
in a production setting, see http://flask.pocoo.org/docs/1.0/deploying/#deployment
for some options.
