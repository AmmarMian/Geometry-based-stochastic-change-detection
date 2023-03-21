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
from utility import (
    generate_covariance_toeplitz,
    sample_complex_gaussian
)
from change_detection import (
    covariance_equality_glrt_gaussian_statistic,
    scale_and_shape_equality_robust_statistic,
    scale_and_shape_equality_robust_statistic_kron,
    scale_and_shape_equality_robust_statistic_sgd,
    scale_and_shape_equality_robust_statistic_sgd_kron
)



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
    n_batches_list = np.unique(
                        np.logspace(1, 3, 5, base=a*b, dtype=int)
                    )
    n_samples = 2*a*b
