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
sys.path.append(os.path.join(os.path.dirname(sys.argv[0]), '../../src'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='darkgrid')
from joblib import Parallel, delayed
from tqdm import trange, tqdm

from utility import (
    generate_covariance_toeplitz,
    sample_complex_gaussian
)
from estimation import (
    tyler_estimator_covariance,
    estimation_cov_kronecker_MM,
    scaledgaussian_mle_natural_gradient_kronecker_fim,
    SCM,
)
from rich import print as rprint

if __name__ == "__main__":

    # Simulation parameters
    # -------------------------------------------------------------------------
    a = 3
    b = 4
    A = generate_covariance_toeplitz(0.3+0.7j, a)
    B = generate_covariance_toeplitz(0.1+0.1j, b)

    # Normalising by the determinant and applying Kronecker structure
    A, B = [M/(np.abs(np.linalg.det(M))**(1/len(M)))
            for M in (A, B)]
    Sigma = np.kron(A, B)

    n_trials = 1000
    n_samples_list = np.unique(np.logspace(1.2, 3, 10, base=a*b, dtype=int))

    # Definition of a single MonteCarlo Trial
    def one_trial(trial_no):
        rng = np.random.default_rng(trial_no)
        results = {'trial_no': trial_no}
        for n_samples in n_samples_list:
            X = sample_complex_gaussian(
                    n_samples, np.zeros((a*b,), dtype=complex),
                    Sigma, random_state=rng
                )
            Sigma_scm = np.cov(X.T)
            A_MM, B_MM, _, _ = estimation_cov_kronecker_MM(X, a, b)
            Sigma_Kron_MM = np.kron(A_MM, B_MM)
            Sigma_Kron_MM = Sigma_Kron_MM/(np.abs(np.linalg.det(Sigma_Kron_MM))**(1/(a*b)))
            Sigma_Tyler, _, _ = tyler_estimator_covariance(X.T)
            Sigma_Tyler = Sigma_Tyler/(np.abs(np.linalg.det(Sigma_Tyler))**(1/(a*b)))

            results[n_samples] = {
                    "SCM": np.linalg.norm(Sigma_scm - Sigma)/np.linalg.norm(Sigma),
                    "Kron MM": np.linalg.norm(Sigma_Kron_MM - Sigma)/np.linalg.norm(Sigma),
                    "Scaled-Gaussian Tyler": np.linalg.norm(Sigma_Tyler - Sigma)/np.linalg.norm(Sigma)

            }
        return results

    # Launching parallel processing
    rprint('[bold]Launching simulation with parameters:')
    print(f'a={a}, b={b}, n_samples={n_samples_list}')
    results = Parallel(n_jobs=-1)(
            delayed(one_trial)(trial_no)
            for trial_no in trange(n_trials)
        )

    # Organising results for plots
    MSE_SCM = np.sum(np.array([[results[trial_no][n_samples]['SCM'] for trial_no in range(n_trials)]
           for n_samples in n_samples_list]), axis=1)
    MSE_Kron_MM = np.sum(np.array([[results[trial_no][n_samples]['Kron MM'] for trial_no in range(n_trials)]
           for n_samples in n_samples_list]), axis=1)
    MSE_Tyler = np.sum(np.array([[results[trial_no][n_samples]['Scaled-Gaussian Tyler'] for trial_no in range(n_trials)]
           for n_samples in n_samples_list]), axis=1)
    plt.figure()
    plt.loglog(n_samples_list, MSE_SCM, label="SCM")
    plt.loglog(n_samples_list, MSE_Kron_MM, label="Kron MM")
    plt.loglog(n_samples_list, MSE_Tyler, label="Scaled-Gaussian Tyler")
    plt.legend()
    plt.xlabel('$N$')
    plt.ylabel('MSE')
    plt.show()
