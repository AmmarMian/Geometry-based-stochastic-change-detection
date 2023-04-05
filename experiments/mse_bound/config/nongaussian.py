
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
        SpecialHermitianPositiveDefinite,
)
from pymanopt.manifolds import StrictlyPositiveVectors
from models import rgrad_scaledgaussian_kronecker


rng = np.random.default_rng(7777)
a = 3
b = 4
n_trials = 10000
n_batches_list = [2, 5, 10, 25, 50, 100, 200, 500, 1000]
n_samples = a*b+1

A = generate_covariance_toeplitz(0.3+0.7j, a)
B = generate_covariance_toeplitz(0.3+0.6j, b)
nu = 1
tau = rng.gamma(nu, scale=1/nu, size=(n_samples, 1))

# Normalising by the determinant and applying Kronecker structure
A, B = [X/(np.abs(np.linalg.det(X))**(1/len(X)))
        for X in (A, B)]


estimators_offline = {
    'MLE scaled gaussian Kronecker':
        lambda X: scaledgaussian_mle_natural_gradient_kronecker_fim_batch(
            np.moveaxis(X, [0, 1], [-1, 1]), a, b),
    # 'MLE scaled gaussian MM Kronecker':
        # lambda X: estimation_cov_kronecker_MM_H0(
            # np.moveaxis(X, [0, 1], [-1, 1]), a, b, return_tau=True,
            # return_info=False)
}

estimators_online = {
    'SGD scaled gaussian Kronecker': {
            'manifold': KroneckerHermitianPositiveScaledGaussian(
                        a, b, n_samples),
            'rgrad': rgrad_scaledgaussian_kronecker,
            'init': (np.eye(a, dtype=complex), np.eye(b, dtype=complex),
                     np.ones((n_samples, 1))),
            'lr': 1/(a*b*n_samples),
        }
}

manifold_A = SpecialHermitianPositiveDefinite(a)
manifold_B = SpecialHermitianPositiveDefinite(b)
manifold_tau = StrictlyPositiveVectors(n_samples)


def mse_function_A(X: np.ndarray, Y: np.ndarray) -> float:
    return manifold_A.dist(X, Y)**2


def mse_function_B(X: np.ndarray, Y: np.ndarray) -> float:
    return manifold_B.dist(X, Y)**2


def mse_function_tau(tau_1: np.ndarray, tau_2: np.ndarray) -> float:
    return float(manifold_tau.dist(tau_1, tau_2)**2)
