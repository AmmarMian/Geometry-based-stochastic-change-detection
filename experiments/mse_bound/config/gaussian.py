
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(sys.argv[0]), '../../src'))
from utility import (
    generate_covariance_toeplitz,
)
from estimation import(
    stochastic_gradient_scaledgaussian,
    stochastic_gradient_scaledgaussian_kronecker,
    scaledgaussian_mle_natural_gradient_fim_batch,
    scaledgaussian_mle_natural_gradient_kronecker_fim_batch,
    estimation_cov_kronecker_MM_H0
)
from manifolds import (
        KroneckerHermitianPositiveScaledGaussian,
        ScaledGaussianFIM,
        rgrad_scaledgaussian,
        rgrad_scaledgaussian_kronecker
)


a = 3
b = 4
A = generate_covariance_toeplitz(0.3+0.7j, a)
B = generate_covariance_toeplitz(0.3+0.6j, b)
nu = None  # To have gaussian data

# Normalising by the determinant and applying Kronecker structure
A, B = [X/(np.abs(np.linalg.det(X))**(1/len(X)))
        for X in (A, B)]

n_trials = 100
n_points_roc = 30
n_batches_list = [2, 5, 10, 25, 50]
batch_no_change = lambda n_batches: int(n_batches/2)
n_samples = a*b+1

estimators_offline = {
    'MLE scaled gaussian': scaledgaussian_mle_natural_gradient_fim_batch,
    'MLE scaled gaussian Kronecker':
        lambda X: scaledgaussian_mle_natural_gradient_kronecker_fim_batch(X, a, b),
    'MLE scaled gaussian MM Kronecker':
        lambda X: estimation_cov_kronecker_MM_H0(X, a, b)
}

estimators_online = {
    'SGD scaled gaussian': {
            'manifold': ScaledGaussianFIM(a*b, n_samples),
            'rgrad': rgrad_scaledgaussian,
            'init': (np.eye(a*b), np.ones(n_samples)),
            'lr': 1
        },
    'SGD scaled gaussian Kronecker': {
            'manifold': KroneckerHermitianPositiveScaledGaussian(
                        a, b, n_samples),
            'rgrad': rgrad_scaledgaussian_kronecker,
            'init': (np.eye(a), np.eye(b), np.ones(n_samples)),
            'lr': 1
        }
}
