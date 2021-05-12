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

if you use ``pip`` requirements will be automatically installed when you
install the ``ramp-*`` packages.

If you use ``conda`` requirements can be manually installed from the
``environment.yml`` file::

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
