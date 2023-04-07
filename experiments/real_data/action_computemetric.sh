#!/bin/bash
# Title: Compute Metric
# Description: Compute noramlised mean ratio and difference between GLRT and SGD.

(
python - << EOF
import numpy as np
import pickle
import os
import re
import sys

if __name__ == "__main__":
    results_dir = "${BASH_ARGV[1]}"
    rx = re.compile('artifact_(.*)pkl')
    search = list(os.walk(results_dir))[0]
    results_files = [file for file in search[-1]
                     if rx.match(file)]

    data = {}
    for file in results_files:
        with open(os.path.join(results_dir, file), 'rb') as f:
            data_file = pickle.load(f)
            data[data_file['name']] = data_file['results']

    # Scaled Gaussian
    glrt = data['scaled_gaussian_glrt']
    sgd = data['scaled_gaussian_sgd']
    ratio = np.abs(np.mean(glrt / sgd))
    diff = np.abs(np.mean(glrt - sgd))
    print('Scaled Gaussian GLRT/SGD ratio: {}'.format(ratio))
    print('Scaled Gaussian GLRT/SGD diff: {}'.format(diff))

    # Scaled Gaussian Kronecker
    glrt = data['scaled_gaussian_kron_glrt']
    sgd = data['scaled_gaussian_kron_sgd']
    ratio = np.abs(np.mean(glrt / sgd))
    diff = np.abs(np.mean(glrt - sgd))
    print('Scaled Gaussian Kronecker GLRT/SGD ratio: {}'.format(ratio))
    print('Scaled Gaussian Kronecker GLRT/SGD diff: {}'.format(diff))
EOF
)
