#!/usr/bin/env bash
set -ex

pip install -r docs/requirements.txt
cd docs
make html
cd ..

set +ex