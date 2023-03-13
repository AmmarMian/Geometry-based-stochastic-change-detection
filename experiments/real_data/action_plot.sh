#!/bin/bash
# Title: Plotting
# Description: Plot Change Detection results raw

python - << EOF
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import re
import sys

if __name__ == "__main__":
    results_dir = "${BASH_ARGV[0]}"
    rx = re.compile('artifact_(.*)pkl')
    search = list(os.walk(results_dir))[0]
    results_files = [file for file in search[-1]
                     if rx.match(file)]

    for file in results_files:
        with open(os.path.join(results_dir, file), 'rb') as f:
            data = pickle.load(f)
        plt.figure(figsize=(9,7))
        plt.imshow(data['results'], aspect='auto', cmap='jet')
        plt.colorbar()
        plt.title(data['name'])
        plt.savefig(os.path.join(results_dir, f'{file[:-3]}.png'))
    plt.show()

EOF
