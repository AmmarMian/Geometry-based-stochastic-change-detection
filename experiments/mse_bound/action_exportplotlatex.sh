#!/bin/bash
# Title: Export plots to LaTeX
# Description: Export MSE plots to pgfplots library thanks to tikzplotlib.

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
import seaborn as sns
sns.set(style="darkgrid")
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
    n_batches_list = [x for x  in data['results_raw'][0] if type(x) == int]
    ICRB_A = lambda n: (config.a**2-1)/(config.b*config.n_samples*n)
    ICRB_B = lambda n: (config.b**2-1)/(config.a*config.n_samples*n)
    ICRB_tau = lambda n: config.n_samples/(n*config.a*config.b)
    for element, icrb in zip(['A', 'B', 'tau'], [ICRB_A, ICRB_B, ICRB_tau]):
        plt.figure()
        plt.title(f'MSE for {element}')
        plt.ylabel(r'$\delta^2$')
        plt.xlabel(r'$n$ batches$')


        for i, results_dict in enumerate(data['results'].values()):
            for j, estimator in enumerate(results_dict):
                plt.loglog(
                    n_batches_list,
                    results_dict[estimator][element],
                    label=f'{estimator}',
                    marker=markers[2*i+j],
                    linestyle='None',
                    markersize=4
                )
        # ICRB
        icrb_element = icrb(np.array(n_batches_list))
        plt.loglog(
            n_batches_list,
            icrb_element,
            label=f'ICRB',
            marker='o',
            linestyle='-',
            markersize=4
        )
        plt.legend()

        print(f"Generating LaTeX file: $RESULTS_DIR/MSE_{element}.tex")
        tikzplotlib.save(
            f"$RESULTS_DIR/MSE_{element}.tex",
            axis_width='0.9\columnwidth',
            axis_height='5cm')
EOF
)
