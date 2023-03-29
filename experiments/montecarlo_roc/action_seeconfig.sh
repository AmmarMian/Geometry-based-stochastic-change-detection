#!/bin/bash
# Title: See experiment config file
# Description: See the config varaibles used for this experiment at the commit when the experiment was launched.

RESULTS_DIR=${BASH_ARGV[0]}
COMMIT_SHA=${BASH_ARGV[1]}

python - << EOF
import pickle
import pydoc
import subprocess

with open('$RESULTS_DIR/artifact.pkl', 'rb') as f:
    data = pickle.load(f)

file = subprocess.check_output(["git", "show", "$COMMIT_SHA:{data['config_file']}"])
pydoc.pager(file)

EOF
