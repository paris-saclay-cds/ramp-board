#!/bin/bash
set -e
flake8 ramp-database/
pytest -rvsl ramp-database/ramp_database --cov=ramp_database --cov-report=term-missing --cov-append