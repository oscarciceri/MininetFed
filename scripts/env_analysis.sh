#!/bin/bash

python3 -m venv env_analysis

. env_analysis/bin/activate && pip install -r analysis/requirements.txt