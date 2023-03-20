#!/bin/bash
# Title: Follow progress
# Description: Follow progress of simulation thanks to the log files (ctrl+c to exit).

RESULTS_DIR=${BASH_ARGV[0]}
LOG_FILES=$(find ${RESULTS_DIR}/*.log | sed -e 's/^/"/g' -e 's/$/"/g' | tr '\n' ' ')
eval "tail -f ${LOG_FILES}"
