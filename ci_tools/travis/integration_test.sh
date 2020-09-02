#!/bin/bash
set -e
cd $HOME
mkdir ramp_deployment
cd ramp_deployment
psql -U postgres -c "CREATE USER mrramp WITH PASSWORD 'mrramp';ALTER USER mrramp WITH SUPERUSER;"
createdb --owner=mrramp databoard_test
ramp setup init

echo "flask:
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
    username: mrramp
    password: mrramp
    host: localhost
    port: 5432
    database: databoard_test" > config.yml
ramp database add-user --login admin_user --password password --firstname firstname --lastname lastname --email admin@email.com --access-level admin
ramp setup init-event --name iris_test
echo "ramp:
    problem_name: iris
    event_name: iris_test
    event_title: "Iris event"
    event_is_public: true
worker:
    worker_type: conda
    conda_env: ramp-iris
dispatcher:
    hunger_policy: exit" > events/iris_test/config.yml
ramp setup deploy-event --event-config events/iris_test/config.yml
ramp-database approve-user --login admin_user
ramp-database sign-up-team --event iris_test --team admin_user
ramp database add-submission --event iris_test --team admin_user --submission my_submission --path "$HOME/ramp_deployment/ramp-kits/iris/submissions/random_forest_10_10"
ramp launch dispatcher --event-config events/iris_test/config.yml -vv