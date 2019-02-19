Notes regarding the RAMP workers
================================

The RAMP "worker" is the object responsible of actually running (training and
evaluating) a submission and returning back the results (predictions and
scores).
The workers are typically launched from a loop set up for a specific event on
the server. The worker can then run the submission either locally on the server
or on a remote machine.

The type of worker used in a certain event is specified in the ``worker``
section of the yml configuration file, using the ``worker_type`` key (see
:ref:`deploy-ramp-event`). Additional keys can be required depending on
the worker type.

Available workers:

* The :class:`ramp_engine.local.CondaEnvWorker` will run the submission in
  a specific isolated conda environment. This worker is specified as
  ``worker_type: conda`` and requires an additional key ``conda_env``
  specifying the name of the conda environment.
