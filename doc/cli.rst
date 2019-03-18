###########################
RAMP Command-Line Interface
###########################

RAMP comes with a set of command-lines allowing to:

* deploy your RAMP server and events;
* interact with the database;
* launch web frontend;
* launch the dispatcher and workers.

.. note::
    In the following sections, we are documenting the ``ramp-*`` subcommands.
    Be aware that you can access those subcommands using ``ramp *`` without a
    dash. As an example::

        ramp-database -h

    is equivalent to::

        ramp database -h

    You can find the documentation using::

        ramp -h

    We advise you to use the ``ramp`` commands since it will aggregate the
    available subcommands of RAMP workflow for instance.

Commands to setup your RAMP server or RAMP events
-------------------------------------------------

.. click:: ramp_utils.cli:main
    :prog: ramp-setup
    :show-nested:

Commands to interact with the RAMP database
-------------------------------------------

.. click:: ramp_database.cli:main
    :prog: ramp-database
    :show-nested:

Commands to interact with the RAMP web frontend
-----------------------------------------------

.. click:: ramp_frontend.cli:main
    :prog: ramp-frontend
    :show-nested:

Commands to interact with the RAMP engine
-----------------------------------------

.. click:: ramp_engine.cli:main
    :prog: ramp-launch
    :show-nested:
