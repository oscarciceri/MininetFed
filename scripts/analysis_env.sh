#!/bin/bash

python3 -m venv analysis_env

. analysis_env/bin/activate && pip install -r analysis/requirements.txt