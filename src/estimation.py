'''
File: estimation.py
Created Date: Tuesday February 1st 2022 - 04:03pm
Author: Ammar Mian
Contact: ammar.mian@univ-smb.fr
-----
Last Modified: Mon Feb 21 2022
Modified By: Ammar Mian
-----
Estimation functions
-----
Copyright (c) 2022 Université Savoie Mont-Blanc
'''
import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.special
import logging

from tqdm import tqdm

from manifolds import (
    HermitianPositiveDefinite,
    KroneckerHermitianPositiveScaledGaussian,
    SpecialHermitianPositiveDefinite,
    KroneckerHermitianPositiveElliptical,
    ScaledGaussianFIM
)
from models import (
    negative_log_likelihood_complex_scaledgaussian,
    negative_log_likelihood_complex_scaledgaussian_batches,
    egrad_scaledgaussian,
    egrad_kronecker_scaledgaussian,
    rgrad_scaledgaussian,
    rgrad_scaledgaussian_kronecker 
)

import autograd.numpy as np_a
from autograd import grad
from pymanopt.function import Callable
from pymanopt import Problem
from pymanopt.manifolds import Product
from pymanopt.solvers import SteepestDescent

def SCM(x, *args):
    """ A function that computes the SCM for covariance matrix estimation
            Inputs:
                * x = a matrix of size p*N with each observation along column dimension
            Outputs:
                * Sigma = the estimate"""

    (p, N) = x.shape
    return (x @ x.conj().T) / N

# -----------------------------------------------------------------
# Online estimators
# -----------------------------------------------------------------
def stochastic_gradient_scaledgaussian(X, init=None,
                                    lr=1, verbosity=0,
                                    return_value="Covariance+texture"):
    """Scaled-Gaussian with zero-mean MLE using a natural
    stochastic gradient on the manifold (SHPDxR+) with FIM.

    Parameters
    ----------
    X : array-like of shape (n_batches, n_samples, n_features)
        Data
    init : tuple of two array-like of shape (n_features, n_features) and (n_samples, 1), optional
        Initial guess, by default Identity matrix and vector of ones
    lr : int, optional
        learning rate, by default 1
    verbosity : int
        Level of verbosity
        0 is no vebose, 1 shows progress and cost functions values
    return_value : string, optional
        Either "Covariance+texture", "Covariance+texture + log". By default "Covariance+texture"
        "Covariance+texture": returns the estimates of covariance and texture
        "Covariance+texture + log" : returns the estimates of covariance, texture
                                     and the log of optimization
    Returns
    -------
    Depends on the input value of return_value
    """

    n_batches, n_samples, n_features = X.shape

    if init is None:
        Sigma = np.eye(n_features).astype(complex)
        tau = np.ones((n_samples, 1))
    else:
        Sigma, tau = init

    if verbosity>0:
        pbar = tqdm(total=n_batches)
    if return_value == "Covariance+texture + log":
        cost_function_values = []
        iterations = []

    manifold = ScaledGaussianFIM(n_features, n_samples)
    for n in range(n_batches):
        r_Sigma, r_tau = rgrad_scaledgaussian(X[n], Sigma, tau)

        Sigma_new, tau_new = manifold.retr(
            (Sigma, tau), 
            (-lr/(n+1)*r_Sigma, -lr/(n+1)*r_tau)
        )


        if return_value == "Covariance+texture + log" or verbosity > 0:
            cost_val = negative_log_likelihood_complex_scaledgaussian(
                X[n], Sigma_new, tau_new
            )
        if return_value == "Covariance+texture + log":
            cost_function_values.append(cost_val)
            iterations.append((Sigma_new, tau_new))
        if verbosity > 0:
            pbar.update()
            pbar.set_description(
                '(cost sample=%.2f)' % (cost_val), 
                refresh=True
                )

        Sigma, tau = Sigma_new, tau_new

    if verbosity > 0:
        pbar.close()

    if return_value == "Covariance+texture":
        return Sigma, tau
    else:
        log = {'x': iterations, 'f(x)':cost_function_values}
        return Sigma, tau, log

def stochastic_gradient_scaledgaussian_kronecker(X, a, b, init=None,
                                    lr=1, verbosity=0,
                                    return_value="AB"):
    """Scaled-Gaussian with zero-mean MLE using a natural
    stochastic gradient on the manifold (SHPDxR+) with FIM.

    Parameters
    ----------
    X : array-like of shape (n_batches, n_samples, n_features)
        Data
    init : tuple of two array-like of shape (n_features, n_features) and (n_samples, 1), optional
        Initial guess, by default Identity matrix and vector of ones
    verbosity : int
        Level of verbosity
        0 is no vebose, 1 shows progress and cost functions values
    return_value : string, optional
        Either "Covariance", "AB" or "AB + log". By default "Covariance"
        "Covariance": returns the kronecker product of A and B
        "AB": returns A and B
        "AB + log" : return A, B and the log of optimization
    Returns
    -------
    Depends on the input value of return_value
    """

    n_batches, n_samples, n_features = X.shape
    if a*b != n_features:
        raise AssertionError("Size of matrices imcompatible with data.")

    if init is None:
        A = np.eye(a).astype(complex)
        B = np.eye(b).astype(complex)
        tau = np.ones((n_samples, 1))
    else:
        A, B, tau = init

    if verbosity>0:
        pbar = tqdm(total=n_samples)
    if return_value == "AB + log":
        cost_function_values = []
        iterations = []

    manifold = KroneckerHermitianPositiveScaledGaussian(a,b,n_samples)
    for n in range(n_batches):
        r_A, r_B, r_tau = rgrad_scaledgaussian_kronecker(X[n], A, B, tau)
        
        A_new, B_new, tau_new = manifold.retr(
            (A, B, tau), 
            (-lr/(n+1)*r_A, -lr/(n+1)*r_B, -lr/(n+1)*r_tau)
        )


        if return_value == "AB + log" or verbosity > 0:
            cost_val = negative_log_likelihood_complex_scaledgaussian(
                X[n], np.kron(A_new,B_new), tau_new
            )
        if return_value == "AB + log":
            cost_function_values.append(cost_val)
            iterations.append((A_new, B_new, tau_new))
        if verbosity > 0:
            pbar.update()
            pbar.set_description(
                '(cost sample=%.2f)' % (cost_val), 
                refresh=True
                )

        A, B, tau = A_new, B_new, tau_new
    
    if verbosity > 0:
        pbar.close()

    if return_value == "AB":
        return A, B, tau
    else:
        log = {'x': iterations, 'f(x)':cost_function_values}
        return A, B, tau, log

# -----------------------------------------------------------------
# Natural gradient estimators
# -----------------------------------------------------------------
def scaledgaussian_mle_natural_gradient_fim_batch(
    X, init=None, verbosity=0, return_value="Covariance+texture",
    maxiter=1000):
    """Scaled-Gaussian MLE by using a natural gradient on the
    manifold (SHPDxR+*) with Fisher Information Metric. Version
    with BATCHES of data

    Parameters
    ----------
    X : array-like of shape (n_batches, n_samples, n_features)
        Data
    init : tuple of two array-like of shapes (n_features, n_features),
        and (n_samples,), optional
        Initial guess, by default Identity matrix and vector of ones.
    verbosity : int, optional
        Level of verbosity of pymanopt. By default 0.
    return_value : string, optional
        Either "Covariance+texture", "Covariance+texture + log". By default "Covariance+texture"
        "Covariance+texture": returns the Covariance + texture
        "Covariance+texture + log" : return Covariance + texture and the log of optimization
    Returns
    -------
    Depends on the input value of return_value
    """

    n_batches, n_samples, n_features = X.shape

    if init is None:
        Sigma = np.eye(n_features).astype(complex)
        tau = np.ones((n_samples,1))
        x0 = (Sigma, tau)
    else:
        x0 = init

    manifold = ScaledGaussianFIM(n_features, n_samples)

    @Callable
    def cost(Sigma, tau):
        # res = 0
        # for batch in range(n_batches):
        #     res += negative_log_likelihood_complex_scaledgaussian(X[batch], Sigma, tau)
        # return res/n_batches
        return negative_log_likelihood_complex_scaledgaussian_batches(
            X, Sigma, tau
        )


    @Callable
    def egrad(Sigma, tau):
        res_Sigma = np.zeros_like(Sigma)
        res_tau = np.zeros_like(tau)
        for batch in range(n_batches):
            res = egrad_scaledgaussian(X[batch], Sigma, tau)
            res_Sigma += res[0]
            res_tau += res[1]
        return res_Sigma/n_batches, res_tau/n_batches

    problem = Problem(manifold=manifold, cost=cost,
                    egrad=egrad, verbosity=verbosity)
    logverbosity= 2*int(return_value != "Covariance")
    solver = SteepestDescent(logverbosity=logverbosity, maxiter=maxiter)
    if logverbosity==0:
        Sigma, tau = solver.solve(problem, x=x0)
    else:
        x, log = solver.solve(problem, x=x0)
        Sigma, tau = x

    if return_value == "Covariance+texture":
        return Sigma, tau
    else:
        return Sigma, tau, log
    
def scaledgaussian_mle_natural_gradient_fim(
    X, init=None, verbosity=0, return_value="Covariance"):
    """Scaled-Gaussian MLE by using a natural gradient on the
    manifold (SHPDxR+*) with Fisher Information Metric.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data
    init : tuple of two array-like of shapes (n_features, n_features),
        and (n_samples,), optional
        Initial guess, by default Identity matrix and vector of ones.
    verbosity : int, optional
        Level of verbosity of pymanopt. By default 0.
    return_value : string, optional
        Either "Covariance", "Covariance + log". By default "Covariance"
        "Covariance": returns the Covariance + texture
        "Covariance + log" : return Covariance + texture and the log of optimization
    Returns
    -------
    Depends on the input value of return_value
    """

    n_samples, n_features = X.shape

    if init is None:
        Sigma = np.eye(n_features).astype(complex)
        tau = np.ones((n_samples,1))
        x0 = (Sigma, tau)
    else:
        x0 = init

    manifold = ScaledGaussianFIM(n_features, n_samples)

    @Callable
    def cost(Sigma, tau):
        return negative_log_likelihood_complex_scaledgaussian(X, Sigma, tau)

    @Callable
    def egrad(Sigma, tau):
        return egrad_scaledgaussian(X, Sigma, tau)

    problem = Problem(manifold=manifold, cost=cost,
                    egrad=egrad, verbosity=verbosity)
    logverbosity= 2*int(return_value != "Covariance")
    solver = SteepestDescent(logverbosity=logverbosity)
    if logverbosity==0:
        Sigma, tau = solver.solve(problem, x=x0)
    else:
        x, log = solver.solve(problem, x=x0)
        Sigma, tau = x

    if return_value == "Covariance":
        return Sigma, tau
    else:
        return Sigma, tau, log

def scaledgaussian_mle_natural_gradient_kronecker_fim_batch(
    X, a, b, init=None, verbosity=0, return_value="AB", maxiter=1000):
    """Scaled-Gaussian MLE by using a natural gradient on the
    manifold (SHPDxR+*) with Fisher Information Metric. Version
    with BATCHES of data

    Parameters
    ----------
    X : array-like of shape (n_batches, n_samples, n_features)
        Data
    init : tuple of two array-like of shapes (n_features, n_features),
        and (n_samples,), optional
        Initial guess, by default Identity matrix and vector of ones.
    verbosity : int, optional
        Level of verbosity of pymanopt. By default 0.
    return_value : string, optional
        Either "AB", "AB + log". By default "AB"
        "AB": returns the values of A, B and texture
        "Covariance": returns kron(A, B)
        "AB + log": returns the values of A, B, texture and logs
    Returns
    -------
    Depends on the input value of return_value
    """

    n_batches, n_samples, n_features = X.shape
    if a*b != n_features:
        raise AssertionError("Size of matrices imcompatible with data.")

    if init is None:
        A = np.eye(a).astype(complex)
        B = np.eye(b).astype(complex)
        tau = np.ones((n_samples, 1))
        x0 = (A, B, tau)
    else:
        x0 = init

    manifold = KroneckerHermitianPositiveScaledGaussian(a, b, n_samples)

    @Callable
    def cost(A, B, tau):
        # res = 0
        # for batch in range(n_batches):
        #     res += negative_log_likelihood_complex_scaledgaussian(X[batch], Sigma, tau)
        # return res/n_batches
        return negative_log_likelihood_complex_scaledgaussian_batches(
            X, np.kron(A,B), tau
        )

    @Callable
    def egrad(A, B, tau):
        res_A = np.zeros_like(A)
        res_B = np.zeros_like(B)
        res_tau = np.zeros_like(tau)
        for batch in range(n_batches):
            res = egrad_kronecker_scaledgaussian(X[batch], A, B, tau)
            res_A += res[0]
            res_B += res[1]
            res_tau += res[2]
        return res_A/n_batches, res_B/n_batches, res_tau/n_batches

    problem = Problem(manifold=manifold, cost=cost,
                    egrad=egrad, verbosity=verbosity)
    logverbosity= 2*int(return_value == "AB + log")
    solver = SteepestDescent(logverbosity=logverbosity)
    if logverbosity==0:
        A, B, tau = solver.solve(problem, x=x0)
    else:
        x, log = solver.solve(problem, x=x0)
        A, B, tau = x

    if return_value == "AB":
        return A, B, tau
    elif return_value == "Covariance":
        return np.kron(A, B), tau
    else:
        return A, B, tau, log


def scaledgaussian_mle_natural_gradient_kronecker_fim(
    X, a, b, init=None, verbosity=0, return_value="Covariance"):
    """Scaled-Gaussian MLE with a Kronecker structure by using a natural
    gradient on the manifold (SHPDxSHPDxR+*) with Fisher Information Metric.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data
    a : int
        Size of matrix A
    b : int
        Size of matrix B
    init : tuple of three array-like of shapes (a, a), (b, b),
        and (n_samples,), optional
        Initial guess, by default Identity matrices and vector of ones.
    verbosity : int, optional
        Level of verbosity of pymanopt. By default 0.
    return_value : string, optional
        Either "Covariance", "AB" or "AB + log". By default "Covariance"
        "Covariance": returns the kronecker product of A and B + texture
        "AB": returns A and B + texture
        "AB + log" : return A, B + texture and the log of optimization
    Returns
    -------
    Depends on the input value of return_value
    """

    n_samples, n_features = X.shape
    if a*b != n_features:
        raise AssertionError("Size of matrices imcompatible with data.")

    if init is None:
        A = np.eye(a).astype(complex)
        B = np.eye(b).astype(complex)
        tau = np.ones((n_samples,1))
        x0 = (A, B, tau)
    else:
        x0 = init

    manifold = KroneckerHermitianPositiveScaledGaussian(a, b, n_samples)

    @Callable
    def cost(A, B, tau):
        Sigma = np_a.kron(A, B)
        return negative_log_likelihood_complex_scaledgaussian(X, Sigma, tau)

    @Callable
    def egrad(A, B, tau):
        return egrad_kronecker_scaledgaussian(X, A, B, tau)

    problem = Problem(manifold=manifold, cost=cost,
                    egrad=egrad, verbosity=verbosity)
    logverbosity= 2*int(return_value == "AB + log")
    solver = SteepestDescent(logverbosity=logverbosity)
    if logverbosity==0:
        A, B, tau = solver.solve(problem, x=x0)
    else:
        x, log = solver.solve(problem, x=x0)
        A, B, tau = x

    if return_value == "AB":
        return A, B, tau
    elif return_value == "Covariance":
        return np.kron(A, B), tau
    else:
        return A, B, tau, log

# -----------------------------------------------------------------
# Fixed-point estimators
# -----------------------------------------------------------------
def estimation_cov_kronecker_MM(X, a, b, tol=0.001, iter_max=30,
                                verbosity=False, return_tau=False):
    """A function that computes the MM algorithm for Kronecker structured
    covariance matrices with Tyler cost function as presented in:
    >Y. Sun, n_features. Babu and D. n_features. Palomar,
    >"Robust Estimation of Structured Covariance Matrix for Heavy-Tailed
    >Elliptical Distributions,"
    >in IEEE Transactions on Signal Processing,
    >vol. 64, no. 14, pp. 3576-3590, 15 July15, 2016,
    >doi: 10.1109/TSP.2016.2546222.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Dataset.
    a : int
        Size of matrix A so that covariance is equal to A \kron B
    b : int
        Size of matrix B so that covariance is equal to A \kron B
    tol : float, optional
        stopping criterion, by default 0.001
    iter_max : int, optional
        number max of iterations, by default 30
    verbosity : bool, optional
        show progress of algorithm at each iteration, by default False
    Returns
    -------
    array-like of shape (n_features, n_features)
        estimate of A.
    array-like of shape (n_features, n_features)
        estimate of B.
    float
        final error between two iterations.
    int
        number of iterations done.
    """

    n_samples, n_features = X.shape
    Y = X.T

    if a*b != n_features:
        raise AssertionError(
         f"Matrice size incompatible ({a}*{b} != {n_features})"
        )
    M_i = Y.reshape((b, a, n_samples), order='F')
    delta = np.inf  # Distance between two iterations
    # Initialise estimate to identity
    A = np.eye(a, dtype=Y.dtype)
    i_A = np.linalg.inv(A)
    B = np.eye(b, dtype=Y.dtype)
    iteration = 0

    if verbosity:
        pbar_v = tqdm(total=iter_max)

    # Recursive algorithm
    while (delta > tol) and (iteration < iter_max):
        # Useful values
        i_B = la.inv(B)
        sqrtm_A = sp.linalg.sqrtm(A)
        sqrtm_B = sp.linalg.sqrtm(B)
        isqrtm_A = la.inv(sqrtm_A)
        isqrtm_B = la.inv(sqrtm_B)

        # Update A with eq. (66)
        M = np.zeros((a, a), dtype=Y.dtype)
        M_numerator = np.zeros((a, a, n_samples), dtype=Y.dtype)
        for i in range(n_samples):
            M_numerator[:, :, i] = M_i[:, :, i].T.conj() @ i_B @  M_i[:, :, i]
            M_denominator = np.trace(i_A@M_numerator[:, :, i])
            M += a/n_samples * (M_numerator[:, :, i]/M_denominator)
        A_new = sqrtm_A @ sp.linalg.sqrtm(isqrtm_A @ M @ isqrtm_A) @ sqrtm_A
        delta_A = la.norm(A_new - A, 'fro') / la.norm(A, 'fro')
        A = A_new
        i_A = la.inv(A)

        # Update B with eq. (67)
        M = np.zeros((b, b), dtype=Y.dtype)
        for i in range(n_samples):
            M_numerator_B = M_i[:, :, i] @ i_A @  M_i[:, :, i].T.conj()
            M_denominator = np.trace(i_A@M_numerator[:, :, i])
            M += b/n_samples * (M_numerator_B/M_denominator)
        B_new = sqrtm_B @ sp.linalg.sqrtm(isqrtm_B @ M @ isqrtm_B) @ sqrtm_B
        delta_B = la.norm(B_new - B, 'fro') / la.norm(B, 'fro')
        B = B_new

        # Condition for stopping
        delta = max(delta_A, delta_B)
        iteration += 1

        if verbosity:
            pbar_v.update()
            pbar_v.set_description(f'(err={delta})', refresh=True)

    if iteration == iter_max:
        logging.info('Kronecker MM: recursive algorithm did not converge')

    # TODO: Understand why we estimate A.T and not A...
    if return_tau:
        return A.T, B, tol, iteration, M_denominator/a*b
    else:
        return A.T, B, tol, iteration

def estimation_cov_kronecker_MM_H0(X, a, b, tol=0.001, iter_max=30,
                                verbosity=False, return_tau=False, return_info=True):
    """A function that computes the MM algorithm for Kronecker structured
    covariance matrices with Tyler cost function as presented in:
    >Y. Sun, n_features. Babu and D. n_features. Palomar,
    >"Robust Estimation of Structured Covariance Matrix for Heavy-Tailed
    >Elliptical Distributions,"
    >in IEEE Transactions on Signal Processing,
    >vol. 64, no. 14, pp. 3576-3590, 15 July15, 2016,
    >doi: 10.1109/TSP.2016.2546222.
    Parameters
    ----------
    X : array-like of shape (T, n_samples, n_features)
        Dataset.
    a : int
        Size of matrix A so that covariance is equal to A \kron B
    b : int
        Size of matrix B so that covariance is equal to A \kron B
    tol : float, optional
        stopping criterion, by default 0.001
    iter_max : int, optional
        number max of iterations, by default 30
    verbosity : bool, optional
        show progress of algorithm at each iteration, by default False
    Returns
    -------
    array-like of shape (n_features, n_features)
        estimate of A.
    array-like of shape (n_features, n_features)
        estimate of B.
    float
        final error between two iterations.
    int
        number of iterations done.
    """

    T, n_samples, n_features = X.shape
    Y = X.T

    if a*b != n_features:
        raise AssertionError(
         f"Matrice size incompatible ({a}*{b} != {n_features})"
        )
    M_i = Y.reshape((b, a, n_samples,T), order='F')
    delta = np.inf  # Distance between two iterations
    # Initialise estimate to identity
    A = np.eye(a, dtype=Y.dtype)
    i_A = np.linalg.inv(A)
    B = np.eye(b, dtype=Y.dtype)
    iteration = 0

    if verbosity:
        pbar_v = tqdm(total=iter_max)

    # Recursive algorithm
    while (delta > tol) and (iteration < iter_max):
        # Useful values
        i_B = la.inv(B)
        sqrtm_A = sp.linalg.sqrtm(A).astype(np.complex128)
        sqrtm_B = sp.linalg.sqrtm(B).astype(np.complex128)
        isqrtm_A = la.inv(sqrtm_A)
        isqrtm_B = la.inv(sqrtm_B)

        # Update A with eq. (66)
        M = np.zeros((a, a), dtype=Y.dtype)
        M_numerator = np.zeros((a, a, n_samples,T), dtype=Y.dtype)
        for t in range(T):
            for i in range(n_samples):
                M_numerator[:, :, i,t] = M_i[:, :, i,t].T.conj() @ i_B @  M_i[:, :, i,t]
                M_denominator = np.trace(i_A@M_numerator[:, :, i,t])
                M += a/(T*n_samples) * (M_numerator[:, :, i,t]/M_denominator)
        A_new = sqrtm_A @ sp.linalg.sqrtm(isqrtm_A @ M @ isqrtm_A) @ sqrtm_A
        delta_A = la.norm(A_new - A, 'fro') / la.norm(A, 'fro')
        A = A_new.astype(np.complex128)
        i_A = la.inv(A)

        # Update B with eq. (67)
        M = np.zeros((b, b), dtype=Y.dtype)
        for t in range(T):
            for i in range(n_samples):
                M_numerator_B = M_i[:, :, i,t] @ i_A @  M_i[:, :, i,t].T.conj()
                M_denominator = np.trace(i_A@M_numerator[:, :, i,t])
                M += b/(T*n_samples) * (M_numerator_B/M_denominator)
        B_new = sqrtm_B @ sp.linalg.sqrtm(isqrtm_B @ M @ isqrtm_B) @ sqrtm_B
        delta_B = la.norm(B_new - B, 'fro') / la.norm(B, 'fro')
        B = B_new.astype(np.complex128)

        # Condition for stopping
        delta = max(delta_A, delta_B)
        iteration += 1

        if verbosity:
            pbar_v.update()
            pbar_v.set_description(f'(err={delta})', refresh=True)

    if iteration == iter_max:
        logging.info('student_t_estimator_covariance_mle: recursive algorithm did not converge')

    if return_tau:
        if return_info:
            return A.T, B, tol, iteration, M_denominator/a*b
        else:
            return A.T, B, M_denominator/a*b
    else:
        if return_info:
            return A.T, B, tol, iteration
        else:
            return A.T, B


def tyler_estimator_covariance_matandtext(X, tol=0.0001, iter_max=20, return_tau=False):
    """ A function that computes the Modified Tyler Fixed Point Estimator for 
    covariance matrix estimation under problem MatAndText.
        Inputs:
            * X = a matrix of size p*N*T with each saptial observation along column dimension and time
                observation along third dimension.
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * Sigma = the estimate
            * delta = the final distance between two iterations
            * iteration = number of iterations til convergence """

    (p, N, T) = X.shape
    delta = np.inf # Distance between two iterations
    Sigma = np.eye(p) # Initialise estimate to identity
    iteration = 0

    # Recursive algorithm
    while (delta>tol) and iteration < iter_max:

        # Compute the textures for each pixel using all the dates avalaibe
        tau = 0
        iSigma = np.linalg.inv(Sigma)
        for t in range(0, T):
            tau = tau + np.diagonal(X[:,:,t].conj().T@iSigma@X[:,:,t])

        # Computing expression of the estimator
        Sigma_new = 0
        for t in range(0, T):
            X_bis = X[:,:,t] / np.sqrt(tau)
            Sigma_new = Sigma_new + (p/N) * X_bis@X_bis.conj().T

        # Imposing trace constraint: Tr(Sigma) = p
        Sigma_new = p*Sigma_new/np.trace(Sigma_new)

        # Condition for stopping
        delta = np.linalg.norm(Sigma_new - Sigma, 'fro') / np.linalg.norm(Sigma, 'fro')

        # Updating Sigma
        Sigma = Sigma_new
        iteration = iteration + 1

    # if iteration == iter_max:
    #     warnings.warn('Recursive algorithm did not converge')
    if return_tau:
        return (Sigma, tau, delta, iteration)
    else:
        return (Sigma, delta, iteration)

def tyler_estimator_covariance(𝐗, tol=0.001, iter_max=20, return_tau=False):
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
        Inputs:
            * 𝐗 = a matrix of size p*N with each observation along column dimension
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * 𝚺 = the estimate
            * δ = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation
    (p,N) = 𝐗.shape
    δ = np.inf # Distance between two iterations
    𝚺 = np.eye(p) # Initialise estimate to identity
    iteration = 0

    # Recursive algorithm
    while (δ>tol) and (iteration<iter_max):
        
        # Computing expression of Tyler estimator (with matrix multiplication)
        τ = np.diagonal(𝐗.conj().T@np.linalg.inv(𝚺)@𝐗)
        𝐗_bis = 𝐗 / np.sqrt(τ)
        𝚺_new = (p/N) * 𝐗_bis@𝐗_bis.conj().T

        # Imposing trace constraint: Tr(𝚺) = p
        𝚺_new = p*𝚺_new/np.trace(𝚺_new)

        # Condition for stopping
        δ = np.linalg.norm(𝚺_new - 𝚺, 'fro') / np.linalg.norm(𝚺, 'fro')
        iteration = iteration + 1

        # Updating 𝚺
        𝚺 = 𝚺_new

    # if iteration == iter_max:
    #     warnings.warn('Recursive algorithm did not converge')
    if return_tau:
        return (𝚺, τ, δ, iteration)
    else:
        return (𝚺, δ, iteration)

