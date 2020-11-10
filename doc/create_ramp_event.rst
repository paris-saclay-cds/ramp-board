Set up a new RAMP event
=======================

.. _deploy-ramp-event:

Deploy a specific RAMP event
----------------------------

Now we want to deploy a specific problem and event. We will demonstrate by
by deploying an ``iris`` event.

First, you need to get the starting kit and the data into the
``ramp_deployment`` directory::

    ~/ramp_deployment $ mkdir ramp-kits
    ~/ramp_deployment $ mkdir ramp-data
    ~/ramp_deployment $ git clone https://github.com/ramp-kits/iris ramp-kits/iris
    ~/ramp_deployment $ git clone https://github.com/ramp-data/iris ramp-data/iris
    ~/ramp_deployment $ cd ramp-data/iris
    ~/ramp_deployment/ramp-data/iris $ python prepare_data.py
    ~/ramp_deployment/ramp-data/iris $ cd ../..

Next, you need to create a configuration file for a specific event. You can
create this configuration file by executing the following command line from
the ``ramp_deployment`` folder::

    ~/ramp_deployment $ ramp setup init-event --name iris_test

The above creates a ``events/iris_test`` directory inside the deployment
directory, and populates it with a ``config.yml`` with the configuration
specific to the event.

The ``config.yml`` file should look like::

    ramp:
        problem_name: iris
        event_name: iris_test
        event_title: "Iris event"
        event_is_public: true
    worker:
        worker_type: conda
        conda_env: ramp-iris
    dispatcher:
        hunger_policy: sleep
        n_workers: 2
        n_threads: 2
        time_between_collection: 1

**ramp configuration**

* ``problem_name``: All events using the same problem should have the same
  ``problem_name``.
* ``event_name``: Must always be preceded with '<problem_name>_' as in
  the example above.
* ``event_title``: Human readable event name to display on website.
* ``event_is_public``: Boolean indication whether or not the event should be
  public.

**worker configuration**

* ``worker_type``: Refer to the documentation about
  :ref:`workers <all_workers>` for more information on worker types available.
* ``conda_env``: Name of the conda environment to use. If not specified, the
  base environment will be used. Only relevant if using a conda worker.

.. _dispatcher_configuration:

**dispatcher configuration**

* ``hunger_policy``: Policy to apply in case that there is no anymore workers
  to be processed. One of `None`, 'sleep' or 'exit', default is `None`:

    * if `None`: the dispatcher will work without interruption;
    * if 'sleep': the dispatcher will sleep for 5 seconds before to check
      for new submission;
    * if 'exit': the dispatcher will stop after collecting the results of
      the last submissions.

* ``n_workers``: Maximum number of workers that can run submissions
  simultaneously.
* ``n_threads``: The number of threads that each worker can use.
* ``time_between_collection``: How long, in seconds, the worker should wait
  before re-checking if the submission is ready for collection. The default is
  1 second.

Before you continue make sure that:

    1.  Your private and public data are stored in the correct folders.

        For example, your data might be in:

        - public data: `ramp_deployment/ramp-kits/<problem_name>/data/`, and
        - private data: `ramp_deployment/ramp-data/<problem_name>/data/`.

        Note:

        - for more information on directory structure of the starting kits see
          `Overall directory structure
          <https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/dev/workflow.html#overall-directory-structure>`_

    2.  The conda environment (set in the ``config.yml`` file above, here
        called `ramp-iris`) used by your event exists. Note, that this only
        applies to events that use the conda worker. To check which conda
        environments are available, use::

        $ conda env list

Now, you can easily deploy the event (adding both problem and event to the
database) by calling from the deployment directory::

    ~/ramp_deployment $ ramp setup deploy-event --event-config events/iris_test/config.yml --no-cloning

Without passing ``--no-cloning``, it will try to clone the starting kit and
data from the https://github.com/ramp-kits/ GitHub organization. If the
starting kit is located on this GitHub organization, you can skip the first
step above, which manually clones the kit and prepares the data, and let
``ramp setup deploy-event`` do this for you.

.. _launch_dispatcher:

Launch the dispatcher to train and evaluate submissions
-------------------------------------------------------

At this stage, you can launch the RAMP dispatcher from the ``ramp_deployment``
directory, which will be in charge of training, evaluating submissions, and
updating the database::

    ~/ramp_deployment $ ramp launch dispatcher --event-config events/iris_test/config.yml -vv

If you are running the dispatcher on a remote server, you want to launch it
within a terminal multiplexer as ``screen`` or ``tmux``. It will allow you
to detach the process and let it run. Refer to the documentation of ``screen``
or ``tmux`` to use them.


Launch several dispatchers at once
----------------------------------

In case that you are running multiple events in parallel, you will want to
start several dispatchers, on for each open event. We provide a daemon which
will be in charge of managing the pool of dispatchers. You can start it as::

    ~/ramp_deployment $ ramp launch daemon --events-dir events --verbose

To can interrupt the daemon by pressing the combination of keyboard keys
`Ctrl+C`. You can start launch the daemon within `tmux` or `screen` as well.
