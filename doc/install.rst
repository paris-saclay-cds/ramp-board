.. _install:

#######
Install
#######

Install from PyPI
=================

The RAMP bundle is available in PyPI and can be installed via `pip`::

    pip install ramp-database ramp-engine ramp-frontend ramp-utils

All dependencies will be installed automatically.

Install from sources
====================

Prerequisites
-------------

The dependencies required by the RAMP bundle are the following:

1. ``ramp-database``:
    * bcrypt
    * click
    * gitpython
    * nbconvert
    * numpy
    * pandas
    * psycopg2-binary
    * sqlalchemy
2. ``ramp-engine``:
    * click
    * numpy
    * psycopg2-binary
    * sqlalchemy
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
4. ``ramp-utils``:
    * click
    * pandas
    * pyyaml

You can install those requirements through ``pip``, using the
``requirements.txt`` file present in the root folder of the ``ramp-board``
repository::

    pip install -r requirements.txt

You can use ``conda`` instead of ``pip`` using the ``environment.yml`` file::

    conda env update --file environment.yml --name <your_env>

The above command updates the <your_env> environment. If you want to keep the
code to run ramp in a separate environment, you can also do::

    conda env create -f environment.yml

This will create the ``testenv`` environment. You can modify the name of the
environment by editing the ``environment.yml`` file. If you go this way,
you need to remember to ``conda activate testenv`` each time when interacting
with ramp or the database.

Install
-------

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

If you want to install all packages in development ("editable") mode, you
can use::

    make inplace
