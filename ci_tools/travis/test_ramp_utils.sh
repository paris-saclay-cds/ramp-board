#!/bin/bash
set -e
flake8 ramp-utils/
pytest -rvsl ramp-utils/ramp_utils --cov=ramp_utils --cov-report=term-missing --cov-append
