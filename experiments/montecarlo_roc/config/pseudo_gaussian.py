
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(sys.argv[0]), '../../src'))
from utility import (
    generate_covariance_toeplitz,
)
from change_detection import (
    scale_and_shape_equality_robust_statistic,
    scale_and_shape_equality_robust_statistic_kron,
    scale_and_shape_equality_robust_statistic_sgd,
    scale_and_shape_equality_robust_statistic_sgd_kron,
    covariance_equality_glrt_gaussian_statistic
)


a = 3
b = 4
A_H0 = generate_covariance_toeplitz(0.3+0.7j, a)
A_H1 = generate_covariance_toeplitz(0.3+0.5j, a)
B_H0 = generate_covariance_toeplitz(0.3+0.6j, b)
B_H1 = generate_covariance_toeplitz(0.4+0.5j, b)
nu_0 = 100
nu_1 = 100

# Normalising by the determinant and applying Kronecker structure
A_H0, A_H1, B_H0, B_H1 = [
            X/(np.abs(np.linalg.det(X))**(1/len(X)))
            for X in (A_H0, A_H1, B_H0, B_H1)
        ]

n_trials = 100
n_points_roc = 100
n_batches_list = [2, 5, 10, 25, 50, 100, 200]
batch_no_change = lambda n_batches: int(n_batches/2)
n_samples = a*b+1

statistics = {
    'Gaussian GLRT': {
        'function': covariance_equality_glrt_gaussian_statistic,
        'args': 'log'
        },
    'Scaled Gaussian GLRT': {
            'function': scale_and_shape_equality_robust_statistic,
            'args': (1e-4, 30, 'log')
        },
    'Scaled Gaussian Kronecker GLRT': {
            'function': scale_and_shape_equality_robust_statistic_kron,
            'args': (a, b)
        },
    'Scaled Gaussian SGD': {
            'function': scale_and_shape_equality_robust_statistic_sgd,
            'args': "Fixed-point"
        },
    'Scaled Gaussian Kronecker SGD': {
            'function': scale_and_shape_equality_robust_statistic_sgd_kron,
            'args': (a, b)
        },
    }

