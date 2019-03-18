Set up a new RAMP event
=======================

.. _deploy-ramp-event:

Deploy a specific RAMP event
----------------------------

Now we will want to deploy a specific problem and event. We will take an
example by deploying an ``iris`` event.

First, you need to get the starting kit and the data::

    mkdir ramp-kits
    mkdir ramp-data
    git clone https://github.com/ramp-kits/iris ramp-kits/iris
    git clone https://github.com/ramp-data/iris ramp-data/iris
    cd ramp-data/iris
    python prepare_data.py
    cd ../..

Next, you need to create a configuration file for a specific event. You can
create this with::

    ramp setup init-event --name iris_test


The above creates a ``events/iris_test`` directory inside the deployment
directory, and populates it with a ``config.yml`` with the configuration
specific to the event.

This config file should look like::

    ramp:
        problem_name: iris
        event_name: iris_test
        event_title: "Iris event"
        event_is_public: true
    worker:
        worker_type: conda
        conda_env: ramp-iris

.. note::
    In the previous configuration example, we are using a conda worker. You can
    refer to the documentation about the :ref:`workers <all_workers>` and
    more precisely the :ref:`conda workers <conda_env_worker>` to have more
    information.

Finally, you can easily deploy the event (adding both problem and event to the
database) by calling from the deployment directory::

    ramp setup deploy-event --event-config events/iris_test/config.yml --no-cloning

Without passing ``--no-cloning``, it will try to clone the starting kit and
data from the https://github.com/ramp-kits/ github organization. If the
starting kit is located there, you can skip the first step of above of manually
cloning the kit and data and preparing the data, and let ``ramp setup
deploy-event`` do that for you.

Launch the dispatcher to train and evaluate submissions
-------------------------------------------------------

At this stage, you can launch the RAMP dispatcher which will be in charge of
training, evaluating submissions, and updating the database::

    ramp launch dispatcher --event-config events/iris_test/config.yml --hunger-policy sleep -vv
