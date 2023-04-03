#!/bin/bash
# Title: Plotting
# Description: Plot Change Detection results raw

python - << EOF
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

if __name__ == "__main__":
    results_dir = "${BASH_ARGV[1]}"
    with open(f'{results_dir}/artifact.pkl', 'rb') as f:
        data = pickle.load(f)

    for n_batches in data['results']:
        plt.figure(figsize=(9,7))
        for statistic_name in data['results'][n_batches]:
            plt.plot(data['results'][n_batches][statistic_name]['pfa'], data['results'][n_batches][statistic_name]['pd'], label=statistic_name)
        plt.title(f'n_batches = {n_batches}')
        plt.legend()
        plt.savefig(os.path.join(results_dir, f'T{n_batches}.png'))
    plt.show()

EOF
