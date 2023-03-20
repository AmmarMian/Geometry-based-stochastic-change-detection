# ========================================
# FileName: functions.py
# Date: 16 mars 2023 - 11:58
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: Some useful functions
# =========================================

import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import multivariate_normal


def generate_covariance_toeplitz(rho, dim, dtype=np.complex64):
    """Generate a toeplitz structured covariance matrix"""
    cov = toeplitz(np.power(rho, np.arange(0, dim))).astype(dtype)
    return cov


def multivariate_complex_normal_samples(mean, covariance, N, pseudo_covariance=0):
    """ A function to generate multivariate complex normal vectos as described in:
        Picinbono, B. (1996). Second-order complex random vectors and normal
        distributions. IEEE Transactions on Signal Processing, 44(10), 2637–2640.
        Inputs:
            * mean = vector of size p, mean of the distribution
            * covariance = the covariance matrix of size p*p(Gamma in the paper)
            * pseudo_covariance = the pseudo-covariance of size p*p (C in the paper)
                for a circular distribution omit the parameter
            * N = number of Samples
        Outputs:
            * Z = Samples from the complex Normal multivariate distribution, size p*N"""

    p, _ = covariance.shape
    Gamma = covariance
    C = pseudo_covariance

    # Computing elements of matrix Gamma_2r
    Gamma_x = 0.5 * np.real(Gamma + C)
    Gamma_xy = 0.5 * np.imag(-Gamma + C)
    Gamma_yx = 0.5 * np.imag(Gamma + C)
    Gamma_y = 0.5 * np.real(Gamma - C)

    # Matrix Gamma_2r as a block matrix
    Gamma_2r = np.block([[Gamma_x, Gamma_xy], [Gamma_yx, Gamma_y]])

    # Generating the real part and imaginary part
    mu = np.hstack((mean.real, mean.imag))
    v = multivariate_normal.rvs(
            mean=mu, cov=Gamma_2r, size=(N, 1)
        ).T
    X = v[0:p, :]
    Y = v[p:, :]
    return X + 1j * Y

