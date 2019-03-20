#!/bin/bash
set -e
flake8 ramp-engine/
pytest -rvsl ramp-engine/ramp_engine --cov=ramp_engine --cov-report=term-missing --cov-append
