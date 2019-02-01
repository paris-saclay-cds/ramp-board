.. _install:

########################
Install and contribution
########################

Prerequisites
=============

The dependencies required by the RAMP bundle are the following:

1. ``ramp-database``:
    * click
    * gitpython
    * ipykernel
    * jupyter
    * numpy
    * pandas
    * psycopg2
    * six
    * sqlalchemy
2. ``ramp-engine``:
    * click
    * numpy
    * psycopg2
    * sqlalchemy
    * six
3. ``ramp-frontend``:
    * bokeh
    * click
    * Flask
    * Flask-Login
    * Flask-Mail
    * Flask-SQLAlchemy
    * Flask-WTF
    * numpy
    * pandas
    * six
4. ``ramp-utils``:
    * bcrypt
    * click
    * pyyaml
    * six

You can install those requirements through ``pip``, using the
``requirements.txt`` file present in the root folder of the ``ramp-board``
repository::

    pip install -r requirements.txt

You can use ``conda`` instead of ``pip`` using the ``environment.yml`` file::

    conda env create -f environment.yml

This will create the ``testenv`` environment. You can modify the name of the
environment by editing the ``environment.yml`` file.

Install
=======

You can install the RAMP bundle of packages by::

    git clone https://github.com/paris-saclay-cds/ramp-board.git
    cd ramp-board
    make install

It will run the install for each RAMP package. You can also install each
package individually::

    cd ramp-database && pip install .
    cd ../ramp-engine && pip install .
    cd ../ramp-frontend && pip install .
    cd ../ramp-utils && pip install .

Test
====

You can run the test using ``pytest`` from the root direcory::

    pytest -vsl .

Contribute
==========

You can contribute to this code through Pull Request on GitHub_. Please, make
sure that your code is coming with unit tests to ensure full coverage and
continuous integration in the API.

.. _GitHub: https://github.com/paris-saclay-cds/ramp-board/pulls
