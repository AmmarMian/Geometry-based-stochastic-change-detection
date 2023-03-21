# ========================================
# FileName: change_detection_roc.py
# Date: 21 mars 2023 - 11:10
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: ROC for a simulated change
# detection scenario.
# =========================================

import sys
import os
sys.path.append(os.path.join(os.path.dirname(sys.argv[0]), '../src'))

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import trange, tqdm

from utility import (
    generate_covariance_toeplitz,
    sample_complex_gaussian
)
from change_detection import (
    Computing_COR_ChangeDetection,
    scale_and_shape_equality_robust_statistic,
    scale_and_shape_equality_robust_statistic_kron,
    scale_and_shape_equality_robust_statistic_sgd,
    scale_and_shape_equality_robust_statistic_sgd_kron
)
from rich import print as rprint

if __name__ == "__main__":

    # Simulation parameters
    # -------------------------------------------------------------------------
    a = 2
    b = 3
    A_H0 = generate_covariance_toeplitz(0.3+0.7j, a)
    A_H1 = generate_covariance_toeplitz(0.7+0.1j, a)
    B_H0 = generate_covariance_toeplitz(0.1+0.1j, b)
    B_H1 = generate_covariance_toeplitz(0.7+0.6j, b)

    # Normalising by the determinant and applying Kronecker structure
    A_H0, A_H1, B_H0, B_H1 = [
                X/(np.abs(np.linalg.det(X))**(1/len(X)))
                for X in (A_H0, A_H1, B_H0, B_H1)
            ]
    Sigma_H0, Sigma_H1 = np.kron(A_H0, B_H0), np.kron(A_H1, B_H1)

    n_trials = 100
    n_points_roc = 20
    n_batches_list = np.unique(np.logspace(0.5, 1.5, 5, dtype=int))
    batch_no_change = lambda n_batches: int(n_batches/2)
    n_samples = 2*a*b

    statistics = {
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

    # Definition of a single MonteCarlo Trial
    def one_trial(trial_no):
        rng = np.random.default_rng(trial_no)

        # To save time, we generate data with the max number of batches
        # then just take the needed batches
        n_batches = max(n_batches_list)
        # Generating data H0
        X_H0_global = np.zeros((a*b, n_samples, n_batches), dtype=complex)
        for batch_no in range(n_batches):
            X_H0_global[..., batch_no] = sample_complex_gaussian(
                        n_samples, np.zeros((a*b,), dtype=complex),
                        Sigma_H0, random_state=rng).T

        # Generating no change data part of H1 scenario
        X_H1_global = np.zeros((a*b, n_samples, batch_no_change(n_batches)), dtype=complex)
        for batch_no in range(batch_no_change(n_batches)):
            X_H1_global[..., batch_no] = sample_complex_gaussian(
                        n_samples, np.zeros((a*b,), dtype=complex),
                        Sigma_H0, random_state=rng).T

        results = {'trial_no': trial_no}
        for n_batches in n_batches_list:
            # Organising data from the corresponding number of batches
            X_H0 = X_H0_global[..., :n_batches]
            X_H1 = np.zeros((a*b, n_samples, n_batches), dtype=complex)
            X_H1[..., :batch_no_change(n_batches)] = X_H1_global[..., :batch_no_change(n_batches)]
            for batch_no in range(batch_no_change(n_batches), n_batches):
                X_H1[..., batch_no] = sample_complex_gaussian(
                            n_samples, np.zeros((a*b,), dtype=complex),
                            Sigma_H1, random_state=rng).T

            # Computing statistics values on data
            results[n_batches] = {}
            for X, scenario in zip([X_H0, X_H1], ['H0', 'H1']):
                results[n_batches][scenario] = {}
                for statistic_name in statistics.keys():
                    results[n_batches][scenario][statistic_name] = \
                            statistics[statistic_name]['function'](
                                X, statistics[statistic_name]['args']
                            )
        return results

    # Launching parallel processing
    rprint('[bold]Launching simulation with parameters:')
    print(f'a={a}, b={b}, n_samples={n_samples}, n_batches_list={n_batches_list}')
    results = Parallel(n_jobs=-1)(
            delayed(one_trial)(trial_no)
            for trial_no in trange(n_trials)
        )

    # Organising results for plots
    for n_batches in n_batches_list:
        plt.figure()
        for statistic_name in statistics.keys():
            statistic_values_H0 = [results[trial_no][n_batches]['H0'][statistic_name]
                                   for trial_no in range(n_trials)]
            statistic_values_H1 = [results[trial_no][n_batches]['H1'][statistic_name]
                                   for trial_no in range(n_trials)]

            pd, pfa = Computing_COR_ChangeDetection(
                    np.array(statistic_values_H0),
                    np.array(statistic_values_H1),
                    n_points_roc)
            plt.plot(pd, pfa, label=statistic_name)
        plt.legend()
        plt.title(f'n_batches = {n_batches}')
    plt.show()

