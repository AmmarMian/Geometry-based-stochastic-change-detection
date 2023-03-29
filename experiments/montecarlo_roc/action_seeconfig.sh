#!/bin/bash
# Title: See experiment config file
# Description: See the config varaibles used for this experiment at the commit when the experiment was launched.

RESULTS_DIR=${BASH_ARGV[1]}
COMMIT_SHA=${BASH_ARGV[0]}

(
python - << EOF
import pickle
import subprocess

with open('$RESULTS_DIR/artifact.pkl', 'rb') as f:
    data = pickle.load(f)

subprocess.run(["git", "show", f"$COMMIT_SHA:{data['config_file']}"])

EOF
) | less
