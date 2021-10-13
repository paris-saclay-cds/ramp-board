######################
Useful tips using RAMP
######################

How to restart a failed submission manually
-------------------------------------------

If for some reason, one of the submission failed and you would like to
re-evaluate this submission, you should change the state of this submission.
You can use the following command to change the status of a submission::

    ~ $ ramp database set-submission-state --submission-id <id> --state new

Since the submission was set to ``new``, the RAMP dispatcher will automatically
pick up this submission to train it again.

Running a standalone worker without connection to database
----------------------------------------------------------

It can happen that you want to use a worker to run a submission locally to
reproduce what the dispatcher is doing or just train a local submission.
You can use the command `ramp launch worker` to this regard. What you need
is to provide a configuration file with the information regarding the
submission and the worker. For instance, let's imagine that you cloned the
iris kit in locally and that you want to run one of the submission.

We cloned the iris repository in::

    /home/user/Documents/ramp/iris

Now, we need to provide a configuration file containing the information used
by the worker. For instance, for a conda worker, we will provide the following
`config.yml`::

    ramp:
        problem_name: iris
        event_name: iris_test
        event_title: "Human readable event name to display on website"
        event_is_public: true
        data_dir: /home/user/Documents/ramp/iris
        kit_dir: /home/user/Documents/ramp/iris
        submissions_dir: /home/user/Documents/ramp/iris/submissions
        predictions_dir: /home/user/Documents/ramp/iris/predictions
        logs_dir: /home/user/Documents/ramp/iris/logs
        sandbox_dir: starting_kit
    worker:
        worker_type: conda
        conda_env: ramp-iris

Then, you can launch the worker for the submission `random_forest_10_10` as::

    ~/Documents/ramp/iris $ ramp launch worker --submission random_forest_10_10

And you can check the results in the `logs` folder.

.. _create_conda_env:

Create automatically a conda environment linked to an event
-----------------------------------------------------------

When launching an event, you might need to create a conda environment. We
provide a command-line tool to automatically create the `conda` environment::

    ~/ramp_deployment $ ramp setup create-conda-env --event-config events/iris_test/config.yml

This command will create the `conda` environment and install the packages
using the `environment.yml` which should be located inside the `ramp-kit`
directory of the challenge.

.. _update_conda_env:

Update automatically a conda environment linked to an event
-----------------------------------------------------------

We provide a command-line tool to automatically update the packages of a RAMP
event. You use it as follow::

    ~/ramp_deployment $ ramp setup update-conda-env --event-config events/iris_test/config.yml

This will update all package using `conda` if they were installed with `conda`
and `pip` whenever installed with `pip`.
