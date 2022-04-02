#!/usr/bin/env bash
set -x
set -e

# Install dependencies with miforge
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh
bash Miniforge3-$(uname)-$(uname -m).sh -b -p $MINIFORGE_PATH
export PATH="$MINIFORGE_PATH/bin:$PATH"

# create the environment
conda create --yes -n testenv python=3.8
conda env update -n testenv -f environment.yml
source activate testenv
conda install --yes sphinx=3.5.4 jinja2=3.0.3 sphinx_rtd_theme numpydoc pygraphviz
pip install eralchemy sphinx-click

# Build and install scikit-learn in dev mode
make inplace

# The pipefail is requested to propagate exit code
set -o pipefail && cd doc && make html | tee ~/log.txt

cd -
set +o pipefail
