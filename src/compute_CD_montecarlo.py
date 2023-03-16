# ========================================
# FileName: compute_CD_montecarlo.py
# Date: 28 février 2023 - 16:09
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: Script implementing a monte-carlo
# simulation of a change in a batch of data
# =========================================

import pprint
import os
import argparse
import pickle
import numpy as np
from scipy.stats import multivariate_normal
from numpy.linalg import det
from joblib import Parallel, delayed
from tqdm import trange

from utility import (
    ToeplitzMatrix,
    generate_covariance
)

from change_detection import (
    covariance_equality_glrt_gaussian_statistic,
    scale_and_shape_equality_robust_statistic,
    scale_and_shape_equality_robust_statistic_kron,
    scale_and_shape_equality_robust_statistic_sgd,
    scale_and_shape_equality_robust_statistic_sgd_kron
)


def generate_data_batch(a: int, b: int, n_samples: int,
                        tau: np.ndarray, Sigma: np.ndarray,
                        rng=None) -> np.ndarray:
    """Generate data samples for a single batch.

    Parameters
    ----------
    a : int
        size of matrix A in model Cov = A kron B
    b : int
        size of matrix B in model Cov = A kron B
    n_samples : int
        number of samples in this batch
    tau : array-like of shape (n_samples,)
        vector of textures 
    Sigma : array-like of shape (a*b, a*b)
        covariance matrix of model
    rng : None or Generator
        rng generator

    Returns
    -------
    array-like of shape (a*b, n_samples)
        data at this batch
    """

    X = multivariate_normal.rvs(
            mean=np.zeros((a*b,)), cov=Sigma, size=n_samples
        ) +\
        1j * multivariate_normal.rvs(
            mean=np.zeros((a*b,)), cov=Sigma, size=n_samples
            )
    return X.T * np.sqrt(tau)


def one_trial(trial_no:int, a:int, b:int, n_samples:int, n_batches:int,
              scenario: str, nu_before:float, nu_after:float,
              rho_before:float, rho_after: float,
              list_statistics:list, list_args:list, list_names:list) -> dict:
    """Definition of a a trial in this monte-carlo-simulation. We generate
    batches of data according to the change scenario and statistical parameters
    then compute all the test statistics.

    Parameters
    ----------
    trial_no : int
        trial_no
    a : int
        size of matrix A in model Cov = A kron B
    b : int
        size of matrix B in model Cov = A kron B
    n_samples : int
        number of samples in this batch
    n_batches : int
        number of total batches of data
    scenario : str
        choice between 'change' or 'nochange'
    nu_before : float
        texture gamma parameter before the change
    nu_after : float
        texture gamma parameter after the change,
        used only for scenario 'change'
    rho_before : float
        rho_before
    rho_after : float
        rho_after
    list_statistics : list
        list_statistics
    list_args : list
        list_args
    list_names : list
        list_names

    Returns
    -------
    dict

    """
 
    rng = np.random.default_rng(trial_no)

    # Managing statistical parameters of change
    tau_before = rng.gamma(nu_before)
    Sigma_before = np.kron(
        ToeplitzMatrix(rho_before, a),
        ToeplitzMatrix(rho_before, b)
    )
    if scenario == "change":
        tau_after = rng.gamma(nu_after)
        Sigma_after = np.kron(
            ToeplitzMatrix(rho_after, a),
            ToeplitzMatrix(rho_after, b)
        )
        Sigma_after = ToeplitzMatrix(rho_after, a*b)
    else:
        tau_after = tau_before
        Sigma_after = Sigma_before
    
    # Generating data
    X = np.zeros((a*b, n_samples, n_batches), dtype=complex)
    for batch in range(n_batches):
        if batch < round(n_batches/2):
            X_batch = generate_data_batch(
                    a, b, n_samples, tau_before, Sigma_before, rng=rng
            )
        else:
            X_batch = generate_data_batch(
                    a, b, n_samples, tau_after, Sigma_after, rng=rng
            )
        X[..., batch] = X_batch

    # Computing statistics
    result = {'trial_no': trial_no}
    for statistic, args, name in zip(list_statistics, list_args, list_names):
        result[name] = statistic(X, args)

    return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Monte-carlo simulation of a "
                                     "change in a batch of data")
    parser.add_argument("scenario", help="Type of CD scenario",
                        choices=['change', 'nochange'])
    parser.add_argument("a",
                        help="dimension of matrix A in model cov = A kron B",
                        type=int)
    parser.add_argument("b",
                        help="dimension of matrix B in model cov = A kron B",
                        type=int)
    parser.add_argument("n_samples", help="number of samples at each batch",
                        type=int)
    parser.add_argument("n_batches", help="Number of batches of data",
                        type=int)
    parser.add_argument("nu_before", help="scale factor before change",
                        type=float)
    parser.add_argument("--nu_after", help="scale factor after change, "
                        "only when scenario is change",
                        type=float, required=False, default=None)
    parser.add_argument("rho_before", help="correlation factor before change",
                        type=float)
    parser.add_argument("--rho_after", help="correlation factor after change,"
                        "only when scenario is change",
                        type=float, required=False, default=None)
    parser.add_argument("n_trials", help="Number of trials to perform",
                        type=int, default=100)
    parser.add_argument("results_dir", type=str,
                        help="Directory in which to store the results")
    args = parser.parse_args()

    if args.scenario == "change" and\
       (args.nu_after is None or args.rho_after is None):
        raise AttributeError(
                r'It is necessary to precise the parameters after the change!')

    # Definition of statistics used
    list_statistics = [
        covariance_equality_glrt_gaussian_statistic,
        scale_and_shape_equality_robust_statistic,
        # scale_and_shape_equality_robust_statistic_kron,
        scale_and_shape_equality_robust_statistic_sgd,
        # scale_and_shape_equality_robust_statistic_sgd_kron
    ]
    list_args = [
        'log',
        (1e-4, 30, 'log'),
        # (args.a, args.b),
        'Fixed-point',
        # (args.a, args.b)
    ]
    list_names = [
        'Gaussian GLRT',
        "Scaled Gaussian GLRT",
        # "Scaled Gaussian Kronecker GLRT",
        "Scaled Gaussian SGD",
        # "Scaled Gaussian Kronecker SGD"
    ]

    print("Monte-carlo simulation with args:")
    pprint.pprint(vars(args))

    # Case scenario = change
    if args.scenario == "change":
        # Doing case H0
        print("Doing simulation H0")
        results_H0 = Parallel(n_jobs=-1)(
                delayed(one_trial)(
                    trial_no, args.a, args.b, args.n_samples,
                    args.n_batches, 'nochange', args.nu_before,
                    args.nu_after, args.rho_before, args.rho_after,
                    list_statistics, list_args, list_names
                )
                for trial_no in trange(args.n_trials)
        )

        # Doing case H1
        print("Doing simulation H1")
        results_H1 = Parallel(n_jobs=-1)(
                delayed(one_trial)(
                    trial_no, args.a, args.b, args.n_samples,
                    args.n_batches, 'change', args.nu_before,
                    args.nu_after, args.rho_before, args.rho_after,
                    list_statistics, list_args, list_names
                )
                for trial_no in trange(args.n_trials)
        )

        toSave = vars(args)
        for name, arg in zip(list_names, list_args):
            toSave[name] = {
                'args': arg,
                'result_H0': [x[name] for x in results_H0],
                'result_H1': [x[name] for x in results_H1]
            }

    else:
        raise NotImplementedError(
                f'Sorry, scenario {args.scenario} not implemented yet'
        )

    print(f"Saving results artifact to {args.results_dir}")
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    with open(os.path.join(args.results_dir, 'artifact.pkl'), 'wb') as f:
        pickle.dump(toSave, f)
