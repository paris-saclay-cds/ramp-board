.. _contribute:

########################
Contribute
########################

Thanks for joining us. We are always happy to welcome new RAMP developers 
among us.

To install ramp please fallow <TODO> guide making sure that you are using::

    make inplace


If you want to install all packages in development ("editable") mode, you
can use::

    make inplace

You will also need to create the database 

Test
====

If you are developing for the RAMP package, you will be interested about
testing your new feature. You can run the test using ``pytest`` from the root
direcory::

    pytest -vsl .

The above will only work when the packages were installed in development mode.
In the other case, you can test the individual packages with::

    pytest -vsl --pyargs ramp_utils ramp_database ramp_frontend ramp_engine

Contribute
==========

You can contribute to this code through Pull Request on GitHub_. Please, make
sure that your code is coming with unit tests to ensure full coverage and
continuous integration in the API.

.. _GitHub: https://github.com/paris-saclay-cds/ramp-board/pulls
