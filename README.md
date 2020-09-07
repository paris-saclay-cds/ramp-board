# RAMP bundle

[![Build Status](https://travis-ci.com/paris-saclay-cds/ramp-board.svg?branch=master)](https://travis-ci.com/paris-saclay-cds/ramp-board)
[![codecov](https://codecov.io/gh/paris-saclay-cds/ramp-board/branch/master/graph/badge.svg)](https://codecov.io/gh/paris-saclay-cds/ramp-board)
[![CircleCI](https://circleci.com/gh/paris-saclay-cds/ramp-board.svg?style=svg)](https://circleci.com/gh/paris-saclay-cds/ramp-board)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

The advanced branch contains advanced features and may be less stable than the master. It works with the [advanced branch of ramp-workflow](https://github.com/paris-saclay-cds/ramp-workflow/tree/advanced).

This repository contains the following RAMP modules:

- [`ramp-database`](https://github.com/paris-saclay-cds/ramp-board/tree/master/ramp-database) - RAMP database module
- [`ramp-engine`](https://github.com/paris-saclay-cds/ramp-board/tree/master/ramp-engine) - RAMP runner service
- [`ramp-frontend`](https://github.com/paris-saclay-cds/ramp-board/tree/master/ramp-frontend) - RAMP frontend server
- [`ramp-utils`](https://github.com/paris-saclay-cds/ramp-board/tree/master/ramp-utils) - RAMP shared utilities

The modules can be installed independently but have been added to the same
repository so they can be kept in sync.


### Contents

- [Installation](#installation)
- [Testing](#testing)


Installation
------------

1. Retrieve the main repository

    ```bash
    git clone https://github.com/paris-saclay-cds/ramp-board
    cd ramp-board
    ```

2. Install Python dependencies using `conda` or `pip`

    - with `conda`

      ```bash
      # Make sure you run the latest version of conda
      conda update conda
      # Set up the virtual environment
      conda env create -f environment.yml
      # Activate it
      conda activate testenv
      ```

    - with `pip`

      ```bash
      pip install -r requirements.txt
      ```

3. Install all the elements at once

    ```bash
    make install
    ```

   or each project independently

    ```bash
    cd ramp-<project>
    pip install .
    ```

Testing
-------

You can run the test suite using `pytest`:

```bash
pytest -vsl
```

You can also check each project separately.
