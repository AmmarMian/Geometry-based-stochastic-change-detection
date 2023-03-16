#!/bin/bash
# Title: Plotting
# Description: Plot data

python - << EOF
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

results_dir = "${BASH_ARGV[0]}"

with open(os.path.join(results_dir, 'artifact.pkl'), 'rb') as f:
    data = pickle.load(f)
plt.plot(data)
plt.savefig(os.path.join(results_dir, 'test.png'))

EOF