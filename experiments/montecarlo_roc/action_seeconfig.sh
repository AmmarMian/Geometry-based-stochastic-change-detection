#!/bin/bash
# Title: See experiment config file
# Description: See the config varaibles used for this experiment at the commit when the experiment was launched.

RESULTS_DIR=${BASH_ARGV[1]}
COMMIT_SHA=${BASH_ARGV[0]}

(
python - << EOF
import pickle
import subprocess
import json
import os

if os.path.exists('$RESULTS_DIR/info.json'):
    with open('$RESULTS_DIR/info.json', 'r') as f:
        data = json.load(f)
    config_file = data['arguments']
elif os.path.exists('$RESULTS_DIR/artifact.pkl'):
    with open('$RESULTS_DIR/artifact.pkl', 'rb') as f:
        data = pickle.load(f)
        config_file = data['config_file']
else:
    config_file = None
if config_file is not None:
    subprocess.run(["git", "show", f"$COMMIT_SHA:{data['config_file']}"])
else:
    print('Sorry, config_file not available. Probably will be available after experiment ended.')
EOF
) | less
