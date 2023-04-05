#!/bin/bash
# Title: Export plots to LaTeX
# Description: Export ROC plots to pgfplots library thanks to tikzplotlib.

RESULTS_DIR=${BASH_ARGV[1]}
COMMIT_SHA=${BASH_ARGV[0]}
(
python - << EOF
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import git
import sys
from importlib import import_module
from pathlib import Path

# Activate latex in matplotlib
plt.rcParams['text.usetex'] = True
import tikzplotlib


if __name__ == "__main__":
    with open(f'$RESULTS_DIR/artifact.pkl', 'rb') as f:
        data = pickle.load(f)

    # Getting config file and importing it
    repo = git.Repo(search_parent_directories=True)
    commit = repo.commit('$COMMIT_SHA')
    config_file_content = commit.tree[data['config_file']].data_stream.read().decode('utf-8')
    config_file_temp = os.path.join(os.path.dirname(data['config_file']), 'config_$COMMIT_SHA.py')
    with open(config_file_temp, 'w') as f:
        f.write(config_file_content)
    sys.path.append(os.path.dirname(data['config_file']))
    sys.path.append('./src')
    config = import_module(Path(config_file_temp).stem)
    os.remove(config_file_temp)

    markers = [
        'x', 's', 'd', 'v', '^', '<', '>', 'p', 'h',
        'H', 'D', '8', 'P', 'X', 'o', 'x', 's', 'd', 'v',
        '^', '<', '>', 'p', 'h', 'H', 'D', '8', 'P', 'X'
    ]
    for statistic_name_glrt, statistic_name_sgd in zip(
        ['Scaled Gaussian GLRT', 'Scaled Gaussian Kronecker GLRT'],
        ['Scaled Gaussian SGD', 'Scaled Gaussian Kronecker SGD']):
        fig = plt.figure()
        plt.xlabel(r'\$P_{FA}$')
        plt.ylabel(r'\$P_D$')

        # Plotting GLRT with max n_batches
        pfa_glrt, pd_glrt = data['results'][config.n_batches_list[-1]][statistic_name_glrt]['pfa'],\
                            data['results'][config.n_batches_list[-1]][statistic_name_glrt]['pd']
        plt.plot(
            pfa_glrt, pd_glrt,
            label=f'GLRT T={config.n_batches_list[-1]}',
            marker='o',
            markersize=4, markevery=10
        )

        # Plotting SGD with increasing n_batches
        for i, n_batches in enumerate(config.n_batches_list):
            pfa_sgd, pd_sgd = \
                data['results'][n_batches][statistic_name_sgd]['pfa'],\
                data['results'][n_batches][statistic_name_sgd]['pd']
            plt.plot(
                pfa_sgd, pd_sgd,
                label=f'SGD T={n_batches}',
                marker=markers[i],
                markersize=4, markevery=10
            )
        plt.legend( loc='center left', bbox_to_anchor=(1.05, 0.5),
                    ncol=1, fontsize=6)

        print('Generating LaTeX file: ', f'$RESULTS_DIR/{statistic_name_glrt}_vs_{statistic_name_sgd}.tex')
        tikzplotlib.save(f'$RESULTS_DIR/{statistic_name_glrt}_vs_{statistic_name_sgd}.tex',
                         axis_width='0.9\columnwidth', axis_height='5cm')
EOF
)
