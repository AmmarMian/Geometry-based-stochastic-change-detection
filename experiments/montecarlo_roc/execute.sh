#!/bin/bash
# ========================================
# FileName: execute.sh
# Date: 10 mars 2023 - 12:07
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: Script to launch montecarlo
# simuation for a change detection task
# =========================================

echo "$@"
RESULTS_DIR=${BASH_ARGV[0]}
ARGUMENTS="${@%"$RESULTS_DIR"}"

echo "Sourcing project variables"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../../CONFIG.sh

echo ""
echo "Moving to $PROJECT_PATH"
cd "$PROJECT_PATH"

echo ""
echo "Activating conda environment: $CONDA_ENV"
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

echo ""
SCRIPT="python experiments/montecarlo_roc/compute_roc.py $ARGUMENTS $RESULTS_DIR"
echo "Now launching script: $SCRIPT"
eval $SCRIPT
