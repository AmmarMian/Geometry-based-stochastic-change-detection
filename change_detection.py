##############################################################################
# Functions used to detect a change in the parameters of a SIRV distribution
# Authored by Ammar Mian, 28/09/2018
# e-mail: ammar.mian@centralesupelec.fr
##############################################################################
# Copyright 2018 @CentraleSupelec
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import numpy as np
import time
import numpy.linalg as la
import scipy as sp
import scipy.stats
import scipy.special
import warnings
from models import negative_log_likelihood_complex_scaledgaussian
from estimation import (
    estimation_cov_kronecker_MM, 
    estimation_cov_kronecker_MM_H0,
    tyler_estimator_covariance,
    tyler_estimator_covariance_matandtext,
    scaledgaussian_mle_natural_gradient_fim,
    stochastic_gradient_scaledgaussian,
    stochastic_gradient_scaledgaussian_kronecker,
    scaledgaussian_mle_natural_gradient_fim_batch,
    SCM
)

##############################################################################
# Log likelihood scaled gaussian with det = 1
##############################################################################
def neglogH0(X, Sigma, tau):
    (p, N, T) = X.shape
    result = T*N*np.log(np.abs(la.det(Sigma))) + T*p*np.sum(np.log(tau))
    for t in range(T):
        Q = np.real(
            np.einsum('ij,ji->i',
                np.conjugate(X[:,:,t]).T@la.inv(Sigma), X[:,:,t]
            )
        )
        result+= np.sum(Q/np.squeeze(tau))
    return np.real(result)

def neglogH1(X, Sigma, tau):
    (p, N) = X.shape
    result = N*np.log(np.abs(la.det(Sigma))) + p*np.sum(np.log(tau))
    Q = np.real(
        np.einsum('ij,ji->i',
            np.conjugate(X).T@la.inv(Sigma), X
        )
    )
    result+= np.sum(Q/np.squeeze(tau))
    return np.real(result)

def negative_log_likelihood_H0_kronecker(X, A, B, tau):
    p, n, T = X.shape
    # a, b = len(A), len(B)
    # iSigma = np.kron(
    #             np.linalg.inv(A),
    #             np.linalg.inv(B)
    #     )
    # tau_bis = np.squeeze(tau)

    # L = 0
    # for t in range(0, T):
    #     Q = np.real(
    #             np.einsum('ij,ji->i',
    #                 np.conjugate(X[:,:,t]).T@iSigma, X[:,:,t]
    #             )
    #         )
            
    #     L += Q/tau_bis
    # return np.real(L.sum()) + n*T*(b*np.log(np.abs(np.linalg.det(A))) 
    #         + a*np.log(np.abs(np.linalg.det(B)))) +\
    #         T*p*np.sum(np.flatten(tau))
    X_bis = X.reshape(p, n*T)
    return negative_log_likelihood_complex_scaledgaussian(
        X_bis.T, np.kron(A, B), tau
    )

def negative_log_likelihood_H1_kronecker(X, list_A, list_B, list_tau):
    L = 0
    for t in range(X.shape[-1]):
        L += negative_log_likelihood_complex_scaledgaussian(
            X[:,:,t].T, np.kron(list_A[t], list_B[t]), list_tau[t]
        )
    return L

##############################################################################
# Statistics using all the data (off-line)
##############################################################################

def covariance_equality_glrt_gaussian_statistic(X, args=None):
    """ GLRT statistic for detecting a change of covariance matrix in a multivariate Gaussian Time Series.
        At each time, Ni.i.d samples are available. A description of the statistic can be found in:
        D. Ciuonzo, V. Carotenuto and A. De Maio, 
        "On Multiple Covariance Equality Testing with Application to SAR Change Detection," 
        in IEEE Transactions on Signal Processing, vol. 65, no. 19, pp. 5078-5091, 1 Oct.1, 2017.
        doi: 10.1109/TSP.2017.2712124
        Inputs:
            * X = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = number of Samples at each date
                * T = length of time series
            * args = None or 'log' to have the log of GLRT
        Outputs:
            * the GLRT statistic given the observations in input"""

    (p, N, T) = X.shape
    S = 0
    logDenominator = 0
    for t in range(0, T):
        St = SCM(X[:, :, t])
        logDenominator = logDenominator + N * np.log(np.abs(np.linalg.det(St)))
        S = S + St / T
    logNumerator = N * T * np.log(np.abs(np.linalg.det(S)))
    if args is not None:
        if args=='log':
            return np.real(logNumerator - logDenominator)
    return np.exp(np.real(logNumerator - logDenominator))

def detectorMLENGLikelihood(X):
    print("Doing estimation with Natural Gradient to do MLE")
    t_beginning = time.time()
    p, N, T = X.shape
    Sigma_H0_MLE, tau_H0_MLE = scaledgaussian_mle_natural_gradient_fim_batch(
        np.swapaxes(X, 0, 2)
    )
    L1 = 0
    for t in range(T):
        Sigma_H1, tau_H1 = scaledgaussian_mle_natural_gradient_fim(X[:,:,t].T)
        L1 += neglogH1(X[:,:,t], Sigma_H1, np.real(tau_H1))

    L0 = neglogH0(X, Sigma_H0_MLE, tau_H0_MLE)
    print(f'Time elapsed : {time.time()-t_beginning} s')
    print('-----------------------------------------------')
    return L0 - L1

def scale_and_shape_equality_robust_statistic(X, args):
    """ GLRT test for testing a change in the scale or/and shape of 
        a deterministic SIRV model.
        Inputs:
            * X = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = number of Samples at each date
                * T = length of time series
            * args = tol, iter_max for Tyler, scale
        Outputs:
            * the statistic given the observations in input"""
    t_beginning = time.time()
    tol, iter_max, scale = args
    (p, N, T) = X.shape

    # Estimating Sigma_0 using all the observations
    (Sigma_0, delta, niter) = tyler_estimator_covariance_matandtext(X, tol, iter_max)
    iSigma_0 = np.linalg.inv(Sigma_0)

    # Some initialisation
    log_numerator_determinant_terms = T*N*np.log(np.abs(np.linalg.det(Sigma_0)))
    log_denominator_determinant_terms = 0
    tau_0 = 0
    logtau_t = 0
    # Iterating on each date to compute the needed terms
    for t in range(0,T):
        # Estimating Sigma_t
        (Sigma_t, delta, iteration) = tyler_estimator_covariance(X[:,:,t], tol, iter_max)

        # Computing determinant add adding it to log_denominator_determinant_terms
        log_denominator_determinant_terms = log_denominator_determinant_terms + \
                                            N*np.log(np.abs(np.linalg.det(Sigma_t)))

        # Computing texture estimation
        tau_0 =  tau_0 + np.diagonal(X[:,:,t].conj().T@iSigma_0@X[:,:,t]) / T
        logtau_t = logtau_t + np.log(np.diagonal(X[:,:,t].conj().T@np.linalg.inv(Sigma_t)@X[:,:,t]))

    # Computing quadratic terms
    log_numerator_quadtratic_terms = T*p*np.sum(np.log(tau_0))
    log_denominator_quadtratic_terms = p*np.sum(logtau_t)

    # Final expression of the statistic
    if scale=='linear':
        lbda = np.exp(np.real(log_numerator_determinant_terms - log_denominator_determinant_terms + \
        log_numerator_quadtratic_terms - log_denominator_quadtratic_terms))
    else:
        lbda = np.real(log_numerator_determinant_terms - log_denominator_determinant_terms + \
        log_numerator_quadtratic_terms - log_denominator_quadtratic_terms)

    #print(f'Time elapsed : {time.time()-t_beginning} s')
    #print('-----------------------------------------------')

    return lbda

def scale_and_shape_glrt(X):
    (p, N, T) = X.shape
    Sigma_H0, _, _  = tyler_estimator_covariance_matandtext(X)
    iSigma_H0 = la.inv(Sigma_H0)

    tau_H0 = 0
    L1 = 0
    for t in range(T):
        tau_H0 =  tau_H0 + np.real(np.diagonal(X[:,:,t].conj().T@iSigma_H0@X[:,:,t])) / (T*p)

        Sigma_H1, _, _ = tyler_estimator_covariance(X[:,:,t])
        tau_H1 = np.diagonal(X[:,:,t].conj().T@la.inv(Sigma_H1)@X[:,:,t])/p
        L1 += neglogH1(X[:,:,t], Sigma_H1, np.real(tau_H1))

    L0 = neglogH0(X, Sigma_H0, tau_H0)
    return L0 - L1

def scale_and_shape_kronecker_glrt(X, args):
    """ GLRT test for Kronecker-shaped matrices. testing a change in the scale or/and shape of 
        a deterministic SIRV model.
        Inputs:
            * X = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = number of Samples at each date
                * T = length of time series
            * args = a, b, tol, iter_max for Kronecker MM, scale
        Outputs:
            * the statistic given the observations in input"""


    a, b, tol, iter_max = args
    (p, N, T) = X.shape

    # H0 estimation
    X_bis = X.reshape(p, N*T)
    #A_H0, B_H0, _, _, tau_H0 = estimation_cov_kronecker_MM(
    #    X_bis.T, a, b, tol, iter_max, return_tau=True
    #)
    (p, N, T) = X.shape
    A_H0, B_H0, tol, iteration, tau_H0  = estimation_cov_kronecker_MM_H0(X.T, a, b, tol=0.001, iter_max=30,
                                verbosity=False, return_tau=True)
    
    Sigma_H0 = np.kron(A_H0,A_H0)
    iSigma_H0 = la.inv(Sigma_H0)

    # H1 estimation
    list_A = []
    list_B = []
    list_tau = []
    tau_H0 = 0
    for t in range(T):
        tau_H0 =  tau_H0 + np.real(
            np.einsum('ij,ji->i',
                np.conjugate(X[:,:,t]).T@iSigma_H0, X[:,:,t]
            )
        )/(T*p)
        A_t, B_t, _, _, tau_t = estimation_cov_kronecker_MM(
            X[:,:,t].T, a, b, tol, iter_max, return_tau=True
        )
        list_A.append(A_t)
        list_B.append(B_t)
        list_tau.append(tau_t)
    L_H1 = negative_log_likelihood_H1_kronecker(X, list_A, list_B, list_tau)
    L_H0 = negative_log_likelihood_H0_kronecker(X, A_H0, B_H0, tau_H0)

    return L_H0-L_H1

def scale_and_shape_equality_robust_statistic_kron(X,args):
    #print("Doing estimation with Kron Fixed-point MLE")
    t_beginning = time.time()
    a, b = args
    (p, N, T) = X.shape
    A, B, tol, iteration, tau2 = estimation_cov_kronecker_MM_H0(X.T, a, b, tol=0.001, iter_max=30,
                                verbosity=False, return_tau=True) # tau est faux !!!!
    Sigma_H0 = np.kron(A,B)
    iSigma_H0 = la.inv(Sigma_H0)

    tau_H0 = 0
    L1 = 0
    for t in range(T):
        tau_H0 =  tau_H0 + np.real(
            np.einsum('ij,ji->i',
                np.conjugate(X[:,:,t]).T@iSigma_H0, X[:,:,t]
            )
        )/(T*p)
        A, B, tol, iteration = estimation_cov_kronecker_MM(X[:,:,t].T, a, b, tol=0.001, iter_max=30,
                                verbosity=False, return_tau=False)
        Sigma_H1 = np.kron(A,B)
        tau_H1 = np.real(
            np.einsum('ij,ji->i',
                np.conjugate(X[:,:,t]).T@la.inv(Sigma_H1), X[:,:,t]
            )
        )/p
        L1 += neglogH1(X[:,:,t], Sigma_H1, np.real(tau_H1))

    L0 = neglogH0(X, Sigma_H0, tau_H0)
    #print(f'Time elapsed : {time.time()-t_beginning} s')
    #print('-----------------------------------------------')
    return L0 - L1


##############################################################################
# Statistics using the data on-line
##############################################################################   

def scale_and_shape_equality_robust_statistic_sgd(X, MLE='Fixed-point'):
    #print("Doing estimation with Stochastic Natural Gradient")
    t_beginning = time.time()
    p, N, T = X.shape
    Sigma_H0_SGD, tau_H0_SGD = stochastic_gradient_scaledgaussian(
        np.swapaxes(X, 0, 2), lr=1/(p*N)
    )
    L1 = 0
    for t in range(T):
        if MLE=='Fixed-point':
            Sigma_H1, _, _ = tyler_estimator_covariance(X[:,:,t])
            tau_H1 = np.real(
                np.einsum('ij,ji->i',
                    np.conjugate(X[:,:,t]).T@la.inv(Sigma_H1), X[:,:,t]
                )
            )/p
        else:
            Sigma_H1, tau_H1 = scaledgaussian_mle_natural_gradient_fim(X[:,:,t].T)
        L1 += neglogH1(X[:,:,t], Sigma_H1, np.real(tau_H1))

    L0 = neglogH0(X, Sigma_H0_SGD, tau_H0_SGD)
    #print(f'Time elapsed : {time.time()-t_beginning} s')
    #print('-----------------------------------------------')
    return L0 - L1

def scale_and_shape_equality_robust_statistic_sgd_kron(X, args):
    #print("Doing estimation with Stochastic Natural Gradient")
    t_beginning = time.time()
    a, b = args
    p, N, T = X.shape
    # Sigma_H0_SGD, tau_H0_SGD = stochastic_gradient_scaledgaussian(
    #     np.swapaxes(X, 0, 2), lr=1/(p*N)
    # )
    A, B, tau_H0_SGD = stochastic_gradient_scaledgaussian_kronecker(np.swapaxes(X, 0, 2), a, b, init=None,
                                    lr=1/(p*N), verbosity=0,
                                    return_value="AB")
    Sigma_H0_SGD = np.kron(A,B)

    L1 = 0
    for t in range(T):
        A, B, tol, iteration = estimation_cov_kronecker_MM(X[:,:,t].T, a, b, tol=0.001, iter_max=30,
                                verbosity=False, return_tau=False)
        Sigma_H1 = np.kron(A,B)
        tau_H1 = np.real(
            np.einsum('ij,ji->i',
                np.conjugate(X[:,:,t]).T@la.inv(Sigma_H1), X[:,:,t]
            )
        )/p
        L1 += neglogH1(X[:,:,t], Sigma_H1, np.real(tau_H1))

    L0 = neglogH0(X, Sigma_H0_SGD, tau_H0_SGD)
    #print(f'Time elapsed : {time.time()-t_beginning} s')
    #print('-----------------------------------------------')
    return L0 - L1

##############################################################################
# COR and Pfa = f(Lambda)
##############################################################################   

def Computing_COR_ChangeDetection(Lambda_H0,Lambda_H1,nPoints):
    """ Compute the COR plots for a detector in a change detection
        problem.
        Inputs:
            * Lambda_H0,Lambda_H1 = (nMC) numpy arrays with:
                * nMC : number of monte carlo
            * args = ???
        Outputs:
            * Pd,Pfa=f(threshold) """

    nMC = Lambda_H0.shape[0]

    Lambda_H0_sort = np.sort(Lambda_H0)
    Lambda_H1_sort = np.sort(Lambda_H1)

    # Thresolds
    Smin = np.min([Lambda_H0_sort[0],Lambda_H1_sort[0]])
    Smax = np.max([Lambda_H0_sort[nMC-1],Lambda_H1_sort[nMC-1]])
    print("Smin,Smax = ",Smin,Smax)
    S = np.logspace(0,np.log10(Smax+Smin),nPoints)

    # Initialisation
    Pd = np.zeros(nPoints)
    Pfa = np.zeros(nPoints)

    # Loop
    for i in range(nPoints):
        Pd[i] = np.sum(Lambda_H1_sort+Smin>S[i])/nMC
        Pfa[i] = np.sum(Lambda_H0_sort+Smin>S[i])/nMC

    return(Pd,Pfa)


def Computing_COR_ChangeDetection_UAVSAR(Lambda_H0,Lambda_H1,nPoints):

    # Computing ROC curves
    number_of_points = 100
    ground_truth = np.load('./Data/ground_truth_uavsar_scene2.npy')
    ground_truth = ground_truth[int(m_r/2):-int(m_r/2), int(m_c/2):-int(m_c/2)]
    pfa_array = np.zeros((number_of_points, len(function_args[0])))
    pd_array = np.zeros((number_of_points, len(function_args[0])))
    for i_s, statistic in enumerate(statistic_names):

        # Sorting values of statistic
        λ_vec = np.sort(vec(results[:,:,i_s]), axis=0)
        λ_vec = λ_vec[np.logical_not(np.isinf(λ_vec))]

        # Selectionning number_of_points values from beginning to end
        indices_λ = np.floor(np.logspace(0, np.log10(len(λ_vec)-1), num=number_of_points))
        λ_vec = np.flip(λ_vec, axis=0)
        λ_vec = λ_vec[indices_λ.astype(int)]

        # Thresholding and summing for each value
        for i_λ, λ in enumerate(λ_vec):
            good_detection = (results[:,:,i_s] >= λ) * ground_truth
            false_alarms = (results[:,:,i_s] >= λ) * np.logical_not(ground_truth)
            pd_array[i_λ, i_s] = good_detection.sum() / (ground_truth==1).sum()
            pfa_array[i_λ, i_s] = false_alarms.sum() / (ground_truth==0).sum()
