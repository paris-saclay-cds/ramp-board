#!/bin/bash
set -e
flake8 ramp-frontend/
pytest -rvsl ramp-frontend --cov=ramp_frontend --cov-report=term-missing --cov-append
