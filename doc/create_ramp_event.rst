Set up a new RAMP event
=======================

.. _deploy-ramp-event:

Deploy a specific RAMP event
----------------------------

FIXME: we should have a quick deployment for this part
Now we will want to deploy a specific event. We will take an example by
deploying an ``iris`` event. First create a folder (e.g. inside the
deployment directory)::

    mkdir events
    cd events
    mkdir iris
    cd iris

FIXME: we should not copy the configuration file. You need to copy the
``ramp_config.yml`` file located in ``ramp-utils/ramp_utils/template`` and
rename it ``config.yml``.

This config file should look like::

    ramp:
        event: iris
        event_name: iris_test
        event_title: "Iris event"
        event_is_public: true
    worker:
        worker_type: conda
        conda_env: ramp-iris

You can easily deploy the event by calling from the deployment directory::

    ramp-utils deploy-ramp-event --event-config events/iris/config.yml

Launch the dispatcher to train and evaluate submissions
-------------------------------------------------------

At this stage, you can launch the RAMP dispatcher which will be in charge of
training, evaluating submissions, and updating the database::

    ramp-launch dispatcher --event-config events/iris/config.yml --hunger-policy sleep -vv
