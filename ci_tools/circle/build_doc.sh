#!/usr/bin/env bash
set -x
set -e

# Install dependencies with miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
   -O miniconda.sh
chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
export PATH="$MINICONDA_PATH/bin:$PATH"
conda update --yes --quiet conda

# create the environment
conda env create --file environment.yml
source activate testenv
conda install --yes sphinx sphinx_rtd_theme numpydoc graphviz
pip install eralchemy sphinx-click

# Build and install scikit-learn in dev mode
make inplace

# The pipefail is requested to propagate exit code
set -o pipefail && cd doc && make html | tee ~/log.txt

cd -
set +o pipefail
