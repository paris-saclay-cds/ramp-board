.. _contribute:

######################
Develop and contribute
######################

Welcome to the RAMP team. We are always happy to have new RAMP developers.

You can contribute to this code by making a `Pull Request
<https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests>`_
on Github_. Please, make sure that your code is coming with unit tests to
ensure full coverage and continuous integration in the API.


.. _GitHub: https://github.com/paris-saclay-cds/ramp-board/pulls


Install for development
-----------------------
To install RAMP please fallow :ref:`install guideline <install>` making sure
that you use the ``make inplace`` option to install RAMP in developer mode.


Prepare database engine
-----------------------
Testing RAMP requires a database cluster, you need to create it similarly as
described in :ref:`create database <set_database>` section.

If you haven't done so already create the ``Postgres database cluster``
using the command::

    ~ $ initdb postgres_dbs

and then start it with::

    ~ $ pg_ctl -D postgres_dbs -l logfile start

Within your database cluster `postgres` database is created automatically.
`pytest` will use it to make and then drop the test engine. But it needs
to know who is the owner of your `postgres` database and therefore you
need to inform RAMP tests about this.
To do this, open the ``db_engine.yml`` file located in the ``ramp-board``
directory.
It should look as follows::

    db_owner: postgres

You need to change <postgres> to the owner of the `postgres` database.

If you don't know who is the owner of your `postgres` database you can
find it out by typing in your terminal::

    ~ $ psql -l

This command will list all of your databases along with their owners.

Test
----
In order to run your tests please create a test conda environment (you will
need to do that only once)::

    ~ $ conda env create -f ci_tools/environment_iris_kit.yml

Also, before running the tests make sure your ``Postgres database cluster`` has
been started. You can always start it using the command::

    ~ $ pg_ctl -D postgres_dbs -l logfile start

where `postgres_dbs` is the database cluster you created in the previous steps.

In addition, you might want to start an SMTP server to run all the tests.
If you don't run the server, some of the tests will fail because they cannot
be run. To launch the server, execute the following
command::

    ~ $ python -m smtpd -n -c DebuggingServer localhost:8025 &

You are now ready to run the tests. You can do so using ``pytest`` from the
root ``ramp-board`` directory::

    ~ $ pytest -vsl .

The above will only work when the packages were installed in development mode.
In the other case, you can test the individual packages with::

    ~ $ pytest -vsl --pyargs ramp_utils ramp_database ramp_frontend ramp_engine


How to contribute
-----------------

This guide is adapted from `scikit-learn contribution guide`_.

.. _scikit-learn contribution guide: https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md


Forking RAMP
============

The preferred way to contribute to RAMP is to fork the `ramp-board repository`_
on GitHub:

.. _ramp-board repository: https://github.com/paris-saclay-cds/ramp-board

1) To fork the `ramp-board repository`_ click on the 'Fork' button near the
   top of the page. This creates a copy of the code under your account
   on the GitHub server.

2) Clone this copy to your local disk::

        $ git clone git@github.com:YourLogin/ramp-board.git
        $ cd ramp-board

3) Create a branch (called 'my-feature' below) to hold your changes::

        $ git checkout -b my-feature

   and start making changes.

.. note::
    Never work in the ``master`` branch!

4) Work on this copy on your computer using Git to do the version
   control. When you're done editing, do::

        $ git add <modified_files>
        $ git commit

   to record your changes in Git, then push them to GitHub with::

        $ git push -u origin my-feature

Finally, go to the web page of your fork of the ramp-board repo,
and click 'Pull request' to send your changes to the maintainers for
review. This will send an email to the committers.

If any of the above seems like magic to you, then look up `Git documentation`_
on the web.

.. _Git documentation: https://git-scm.com/documentation


Contributing Pull Requests
==========================

It is recommended to check that your contribution complies with the
following rules before submitting a pull request:

-  Follow the coding-guidelines_ as for scikit-learn.

-  When applicable, use the validation tools and other code in the
   `ramp_utils` subpackage.

-  If your pull request addresses an issue, please use the title to describe
   the issue and mention the issue number in the pull request description to
   ensure a link is created to the original issue.

-  All public methods should have informative docstrings with sample
   usage presented as doctests when appropriate.

-  Please prefix the title of your pull request with `[MRG]` if the
   contribution is complete and should be subjected to a detailed review.
   Incomplete contributions should be prefixed `[WIP]` to indicate a work in
   progress (and changed to `[MRG]` when it matures). WIPs may be useful to:
   indicate you are working on something to avoid duplicated work, request
   broad review of functionality or API, or seek collaborators. WIPs often
   benefit from the inclusion of a `task list`_ in the PR description.

-  All other tests pass when everything is rebuilt from scratch. On
   Unix-like systems, check with (from the toplevel source folder)::

        $ make

-  Documentation and high-coverage tests are necessary for enhancements
   to be accepted.

-  At least one paragraph of narrative documentation with links to
   references in the literature (with PDF links when possible) and
   the example.

.. _coding-guidelines: http://scikit-learn.org/dev/developers/contributing.html#coding-guidelines
.. _task list: https://github.com/blog/1375-task-lists-in-gfm-issues-pulls-comments

You can also check for common programming errors with the following
tools:

-  Code with good unittest coverage (at least 80%), check with::

        $ pip install pytest pytest-cov
        $ pytest -vsl .

-  No flake8 warnings (which includes pep8 and pyflakes), check with::

        $ pip install flake8
        $ flake8 path/to/module.py

Filing bugs
===========
We use Github issues to track all bugs and feature requests; feel free to
open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the
following rules before submitting:

-  Verify that your issue is not being currently addressed by other
   issues_ or `pull requests`_.

-  Please ensure all code snippets and error messages are formatted in
   appropriate code blocks.
   See `Creating and highlighting code blocks`_.

-  Please include your operating system type and version number, as well
   as your Python, scikit-learn, numpy, and scipy versions. This information
   can be found by runnning the following code snippet::

    python
    import platform; print(platform.platform())
    import sys; print("Python", sys.version)
    import numpy; print("NumPy", numpy.__version__)
    import scipy; print("SciPy", scipy.__version__)
    import sklearn; print("Scikit-Learn", sklearn.__version__)

-  Please include a reproducible_ code snippet or link to a gist_.
   If an exception is raised, please provide the traceback.

.. _Creating and highlighting code blocks: https://help.github.com/articles/creating-and-highlighting-code-blocks
.. _issues: https://github.com/paris-saclay-cds/ramp-board/issues
.. _pull requests: https://github.com/paris-saclay-cds/ramp-board/pulls
.. _reproducible: https://stackoverflow.com/help/mcve
.. _gist: https://gist.github.com

Documentation
=============

We are glad to accept any sort of documentation: function docstrings,
reStructuredText documents (like this one), tutorials, etc.
reStructuredText documents live in the source code repository under the
doc/ directory.

You can edit the documentation using any text editor and then generate
the HTML output by typing ``make html`` from the doc/ directory.
Alternatively, ``make`` can be used to quickly generate the
documentation without the example gallery. The resulting HTML files will
be placed in _build/html/ and are viewable in a web browser. See the
README file in the doc/ directory for more information.

For building the documentation, you will need

    - sphinx_,
    - sphinx_rtd_theme_,
    - numpydoc_,
    - graphviz_,
    - eralchemy_,
    - sphinx-click_,
    - matplotlib_.

.. _sphinx: http://sphinx-doc.org
.. _matplotlib: https://matplotlib.org
.. _sphinx_rtd_theme: https://sphinx-rtd-theme.readthedocs.io/en/stable/
.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html
.. _graphviz: https://www.graphviz.org/
.. _eralchemy: https://pypi.org/project/ERAlchemy/
.. _sphinx-click: https://sphinx-click.readthedocs.io/en/latest/

When you are writing documentation, it is important to keep a good
compromise between mathematical and algorithmic details, and give
intuition to the reader on what the algorithm does. It is best to always
start with a small paragraph with a hand-waving explanation of what the
method does to the data and a figure (coming from an example)
illustrating it.


Minor release process
---------------------

The following explain the main steps to release `ramp-board`:

1. Run `bumpversion release`. It will remove the `dev0` tag.
2. Commit the change `git commit -am "bumpversion 0.<version number>.0"`
   (e.g., `git commit -am "bumpversion 0.5.0"`).
3. Create a branch for this version (e.g.,
   `git checkout -b 0.<version number>.X`).
4. Push the new branch into the upstream remote ramp-board repository.
5. Create a GitHub release by clicking 'Draft a new release' `here
   <https://github.com/paris-saclay-cds/ramp-board/releases>`_. Copy the
   release notes from `whats_new
   <https://paris-saclay-cds.github.io/ramp-docs/ramp-board/dev/whats_new.html>`_.
6. Change the symlink in the `ramp-docs
   <https://github.com/paris-saclay-cds/ramp-docs>`_ repository such that
   stable points to the latest release version, i.e, 0.<version number>. To do
   this, clone the `ramp-docs` repository, `cd` into `ramp-docs/ramp-board/`
   then run `unlink stable`, followed by
   `ln -s 0.<version number> stable`. To check that
   this was performed correctly, ensure that `ramp-board/stable
   <https://github.com/paris-saclay-cds/ramp-docs/blob/master/ramp-board/stable>`_
   has the new version number.
7. `cd` back into the `ramp-board` code repository and ensure you are in the
   release branch (e.g., branch `0.5.X`). Remove unnecessary files
   with `make clean-dist` then push on PyPI with `make upload-pypi`.
8. Switch to `master` branch and run `bumpversion minor`, commit and push on
   upstream.
9. Add a new `v0.<version number>.rst` file in `doc/whats_new/
   <https://github.com/paris-saclay-cds/ramp-board/tree/master/doc/whats_new>`_
   and `.. include::` this new file in `doc/whats_new.rst
   <https://github.com/paris-saclay-cds/ramp-board/blob/master/doc/whats_new.rst>`_.

Note that the steps 4, 5 and 7 should be performed while in the release
branch, e.g. branch `0.5.X`.

Patch/bug fix release process
-----------------------------

1. Checkout the branch for the lastest release, e.g.,
   `git checkout 0.5.X`.
2. Find the commit(s) hash of the bug fix commit you wish to back port
   using `git log`.
3. Append the bug fix commit(s) to the branch using `git cherry pick <hash>`.
4. Bump the version number with `bumpversion patch`. This will bump the
   patch version, for example from 0.5.0 to 0.5.1.dev0.
5. Mark the current version as release version (as opposed to 'dev' version)
   with `bumpversion release --allow-dirty`. It will bump the version from
   0.5.1.dev0 to 0.5.1.
6. Commit the changes with `git commit -am 'bumpversion <new version>'`.
7. Push the changes to the release branch in upstream, e.g.
   `git push <upstream remote> <release branch>`
8. Remove unnecessary files with `make clean-dist` then push on PyPI with
   `make upload-pypi`.
9. Create a GitHub release by clicking 'Draft a new release' `here
   <https://github.com/paris-saclay-cds/ramp-board/releases>`_. Note down the
   bug fixes added in the patch.
