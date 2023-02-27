'''
File: estimation.py
Created Date: Tuesday February 21st 2022 - 04:05pm
Author: Ammar Mian
Contact: ammar.mian@univ-smb.fr
-----
Last Modified: Mon Feb 21 2022
Modified By: Ammar Mian
-----
Cost functions and associated gradients used in estimation
-----
Copyright (c) 2022 Université Savoie Mont-Blanc
'''

import autograd.numpy as np_a
import autograd.numpy.linalg as la_a
import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.special


from utility import (
    inv, invsqrtm,
    hermitian, proj_shpd
)


# -----------------------------------------------------------------
# Cost functions
# -----------------------------------------------------------------
def negative_log_likelihood_complex_scaledgaussian_batches(X, Sigma, tau):
    """negative log-likelihood of scaled-Gaussian model when there are
    several batches sharing same covariance and texture parameter.

    Parameters
    ----------
    X : array-like of shape (n_batches, n_samples, n_features)
        Data
    Sigma : array-like of shape (n_features, n_features)
        Covariance matrix
    tau : array-like of shape (n_samples,)
        texture vector
    normalization : str, optional
        Normalization type of sigma, either "determinant" or anything else.
    Returns
    -------
    float
        the negative log-likelihood
    """
    n_batches = len(X)
    res = 0
    for batch in range(n_batches):
        res += negative_log_likelihood_complex_scaledgaussian(X[batch], Sigma, tau)
    return res/n_batches


def negative_log_likelihood_complex_scaledgaussian(X, sigma, tau, normalization="determinant"):
    """negative log-likelihood of scaled-Gaussian model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data
    Sigma : array-like of shape (n_features, n_features)
        Covariance matrix
    tau : array-like of shape (n_samples,)
        texture vector
    normalization : str, optional
        Normalization type of sigma, either "determinant" or anything else.
    Returns
    -------
    float
        the negative log-likelihood
    """
    X_bis = X.T
    p, n = X_bis.shape

    # compute quadratic form
    Q = np_a.real(
            np_a.einsum('ij,ji->i',
                np_a.conjugate(X_bis).T@la_a.inv(sigma), X_bis
            )
        )

    tau = np_a.squeeze(tau)
    L = p*np_a.log(tau) + Q/tau
    if normalization == "determinant":
        L = np_a.real(np_a.sum(L))
    else:
        L = n*np_a.real(np_a.linalg.det(sigma)) + np_a.real(np_a.sum(L))

    return L/(n*p)

def cost_function(X, Sigma, tau):
    n_batches, n_samples, n_features = X.shape
    if X.ndim == 3:
        res = 0
        for batch in range(n_batches):
            res += negative_log_likelihood_complex_scaledgaussian(X[batch], Sigma, tau)
        # n_batches, n_samples, n_features = X.shape
        # res = negative_log_likelihood_complex_scaledgaussian(
        #     X.reshape((n_batches*n_samples, n_features)), Sigma, np.tile(tau.flatten(), n_batches)
        #     )
        return res / n_batches
    else:
        return negative_log_likelihood_complex_scaledgaussian(
            X, Sigma, tau
        )

# -----------------------------------------------------------------
# Gradients euclidiens
# -----------------------------------------------------------------
def egrad_scaledgaussian(X, sigma, tau):
    """Euclidean gradient of negative_log_likelihood_complex_scaledgaussian.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data
    Sigma : array-like of shape (n_features, n_features)
        Covariance matrix
    tau : array-like of shape (n_samples,)
        texture vector
    Returns
    -------
    list of arrays of shape [(n_features, n_features), (n_samples,)]
        the Euclidean gradient
    """

    p, n = X.T.shape
    sigma_inv = la.inv(sigma)


    # grad sigma
    X_bis = (X.T) / np.sqrt(tau).T
    grad_sigma = X_bis@X_bis.conj().T
    grad_sigma = -sigma_inv@grad_sigma@sigma_inv

    # grad tau
    X_bis = X.T
    q = np.real(np.einsum('ij,ji->i',
                            np.conjugate(X_bis).T@la.inv(sigma),
                            X_bis))
    q = q.reshape((-1, 1))
    grad_tau = np.real((p*tau-q) * (tau**(-2)))


    return grad_sigma/(n*p), grad_tau/(n*p)

def egrad_kronecker_scaledgaussian(X, A, B, tau):
    """Euclidean gradient of negative_log_likelihood_complex_scaledgaussian
    buth when sigma is subject to Kronecker model:
    $\boldsymbol{\Sigma} = \mathbf{A} \kron \mathbf{B}$.
    n_features=a*b.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data
    A : array-like of shape (a, a)
        Matrix A.
    B : array-like of shape (b, b)
        Matrix B
    tau : array-like of shape (n_samples,)
        texture vector
    Returns
    -------
    list of arrays of shape [(a, a), (b, b), (n_samples,)]
        the Euclidean gradient
    """
    # Pre-calculations
    n, p = X.shape
    a, b = len(A), len(B)
    i_A = inv(A)
    i_B = inv(B)

    # Grad A, B
    grad_A = np.zeros_like(A)
    grad_B = np.zeros_like(B)
    for i in range(n):
        M_i = X[i, :].reshape((b, a), order='F')

        grad_A -= (1/tau[i]) * M_i.T@i_B.conj()@M_i.conj()
        grad_B -= (1/tau[i]) * M_i@i_A.conj()@M_i.T.conj()

    grad_A = (i_A@grad_A@i_A)/(n*p) + (b/p)*i_A
    grad_B = (i_B@grad_B@i_B)/(n*p) + (a/p)*i_B


    # Grad tau
    X_bis = X.T
    q = np.real(
            np.einsum('ij,ji->i',
                np.conjugate(X_bis).T@np.kron(i_A, i_B), X_bis
            )
        )
    q = q.reshape((-1, 1))
    grad_tau = np.real((p*tau-q) * (tau**(-2)))

    return (grad_A, grad_B, grad_tau/(n*p))

# -----------------------------------------------------------------
# Gradients riemanniens
# -----------------------------------------------------------------

def rgrad_scaledgaussian(X, Sigma, tau):
    """Riemannian gradient of negative_log_likelihood_complex_scaledgaussian.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data
    Sigma : array-like of shape (n_features, n_features)
        Covariance matrix
    tau : array-like of shape (n_samples,1)
        texture vector
    Returns
    -------
    list of arrays of shape [(n_features, n_features), (n_samples,)]
        the Riemannian gradient
    """

    # Pre-calculations
    n, p = X.shape
    i_Sigma = inv(Sigma)
    X_bis = X.T
    q = np.real(
            np.einsum('ij,ji->i',
                np.conjugate(X_bis).T@i_Sigma, X_bis
            )
        )

    # Grad tau
    q = q.reshape((n, 1))
    r_tau = -(np.real(q) - p*tau)

    # Grad Sigma
    temp = X_bis / np.sqrt(tau.flatten())
    r_Sigma = temp@temp.T.conj()
    r_Sigma = -proj_shpd(Sigma, r_Sigma)

    #return r_Sigma/(n*p), r_tau.reshape((n,1))/(n*p)
    return p*r_Sigma, n*r_tau.reshape((n,1))

def rgrad_scaledgaussian_kronecker(X, A, B, tau):
    ## Pre-calculations
    n, p = X.shape
    a, b = len(A), len(B)
    if a*b != p:
        raise AssertionError("Size of matrices imcompatible with data.")
    
    i_A = inv(A)
    i_B = inv(B)

    # Grad A, B
    grad_A = np.zeros_like(A)
    grad_B = np.zeros_like(B)
    for i in range(n):
        M_i = X[i, :].reshape((b, a), order='F')

        PA=proj_shpd(A,M_i.T@i_B.conj()@M_i.conj()) #B-T ou B-1
        PB=proj_shpd(B,M_i@i_A.conj()@M_i.T.conj()) # idem
        grad_A -= (1/(b*tau[i])) * PA # ou + ?
        grad_B -= (1/(a*tau[i])) * PB # ou + ?


    # Grad tau
    X_bis = X.T
    q = np.real(
            np.einsum('ij,ji->i',
                np.conjugate(X_bis).T@np.kron(i_A, i_B), X_bis
            )
        )
    q = q.reshape((n, 1))
    grad_tau = -(np.real(q) - p*tau)

    return (p*grad_A, p*grad_B, n*grad_tau)


def rgrad_scaledgaussian_batches(X, Sigma, tau):
    """Riemannian gradient of negative_log_likelihood_complex_scaledgaussian
    when there are several batches with common covariacne and texture parameters.

    Parameters
    ----------
    X : array-like of shape (n_batches, n_samples, n_features)
        Data
    Sigma : array-like of shape (n_features, n_features)
        Covariance matrix
    tau : array-like of shape (n_samples,1)
        texture vector
    Returns
    -------
    list of arrays of shape [(n_features, n_features), (n_samples,)]
        the Riemannian gradient
    """
    n_batches = len(X)
    res_Sigma = np.zeros_like(Sigma)
    res_tau = np.zeros_like(tau)
    for batch in range(n_batches):
        res = rgrad_scaledgaussian(X[batch], Sigma, tau)
        res_Sigma += res[0]
        res_tau += res[1]
    return res_Sigma, res_tau