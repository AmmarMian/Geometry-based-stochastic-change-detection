'''
File: utils.py
Created Date: Tuesday February 1st 2022 - 04:04pm
Author: Ammar Mian
Contact: ammar.mian@univ-smb.fr
-----
Last Modified: Thu Feb 03 2022
Modified By: Ammar Mian
-----
General purpose utils functions/classes
-----
Copyright (c) 2022 Université Savoie Mont-Blanc
'''
import numpy as np
import scipy as sp
from numpy.linalg import det
from scipy.stats import (
    unitary_group, ortho_group,
    multivariate_normal
)
import logging
import autograd.numpy as np_a
import math
from scipy.linalg import toeplitz


def generate_covariance_toeplitz(rho, dim, dtype=np.complex128):
    """Generate a toeplitz structured covariance matrix"""
    cov = toeplitz(np.power(rho, np.arange(0, dim))).astype(dtype)
    return cov


def multivariate_complex_normal_samples(mean, covariance, N, pseudo_covariance=0, rng=None):
    """ A function to generate multivariate complex normal vectos as described in:
        Picinbono, B. (1996). Second-order complex random vectors and normal
        distributions. IEEE Transactions on Signal Processing, 44(10), 2637–2640.
        Inputs:
            * mean = vector of size p, mean of the distribution
            * covariance = the covariance matrix of size p*p(Gamma in the paper)
            * pseudo_covariance = the pseudo-covariance of size p*p (C in the paper)
                for a circular distribution omit the parameter
            * N = number of Samples
            * rng = random generator
        Outputs:
            * Z = Samples from the complex Normal multivariate distribution, size p*N"""

    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

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
            mean=mu, cov=Gamma_2r, size=(N, 1), rng=rng
        ).T
    X = v[0:p, :]
    Y = v[p:, :]
    return np.complex128(X + 1j * Y)



def sqrt_int(X: int):
    """To find matching number of threads row and columns"""
    N = math.floor(math.sqrt(X))
    while bool(X % N):
        N -= 1
    M = X // N
    return M, N

def compute_several_statistics_tomemmap(X:np.ndarray, 
    list_statistics:list, list_args:list, memmap:np.memmap, line:int,
    column:int, flush_line:bool=True)->None:
    """Compute and aggregate test statistic value for several
    test statistics.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        input array where n_samples is the number of pixels and
        n_features, the number of dimensions of the diversity.
    list_statistics : list
        list of functions objects corresponding to the test
        statistics
    list_args : list
        list of arg passed to the test statistic function
    memmap : numpy memmap of shape (n_lines, n_columns, n_statistics)
        object to write on disk directly
    line : int
        line number
    column : int
        column number
    flush_line : bool
        If True, we only flush at end of line to save disk usage
    """

    # checking if not already computed
    if np.any(np.isnan(memmap[line, column])):
        memmap[line, column] = np.array(
            [ statistic(X, arg) 
              for statistic, arg in 
              zip(list_statistics, list_args) ]
        )

        # Write on disk
        if flush_line:
            if column == memmap.shape[1] - 1:
                memmap.flush()
        else:
            memmap.flush()


def vectorize_spatial(X:np.ndarray)->np.ndarray:
    """Vectorize spatial dimensions of SITS.

    Parameters
    ----------
    X : array_like of shape (.., n_lines, n_columns)
        where ... represents any number of dimensions and
        n_lines, n_columns represents the number of lines 
        and columns of each image

    Returns
    -------
    array-like of shape (..., n_lines*n_columns)
        Vectorised image along spatial dimensions
    """
    return X.reshape(X.shape[:-2]+(-1,))

def ToeplitzMatrix(rho, p):
    """ A function that computes a Hermitian semi-positive matrix.
            Inputs:
                * rho = a scalar
                * p = size of matrix
            Outputs:
                * the matrix """

    return sp.linalg.toeplitz(np.power(rho, np.arange(0, p)))

def hermitian(X):
    return .5*(X + np.conjugate(X.T))


def proj_shpd(A, xi):
    iA = invsqrtm(A)
    return hermitian(xi) - np.trace(iA@xi@iA)*A/len(A)


def MSE(Sigma_true, Sigma):
    return np.linalg.norm(Sigma_true-Sigma, 'fro')/np.linalg.norm(Sigma_true, 'fro')


def invsqrtm(M):
    """Inverse sqrtm for a SPD matirx

    Parameters
    ----------
    M : array-like of shape (n_features, n_features)
        input matrix
    """

    eigvals, eigvects = np_a.linalg.eigh(M)
    eigvals = np_a.diag(1/np_a.sqrt(eigvals))
    Out = np_a.dot(
        np_a.dot(eigvects, eigvals), np_a.conjugate(eigvects.T))
    return Out


def inv(M):
    eigvals, eigvects = np_a.linalg.eigh(M)
    eigvals = np_a.diag(1/eigvals)
    Out = np_a.dot(
        np_a.dot(eigvects, eigvals), np_a.conjugate(eigvects.T))
    return Out

def generate_covariance(n_features, unit_det=False,
                    random_state=None, dtype=float):
    """Generate random covariance of size n_features using EVD.

    Parameters
    ----------
    n_features : int
        number of features
    unit_det : bool, optional
        whether to have unit determinant or not, by default False
    random_state : None or a numpy random data generator
        for reproducibility matters, one can provide a random generator
    dtype : float or complex
        dtype of covariance wanted
    """
    if random_state is None:
        rng = np.random
    else:
        rng = random_state

    # Generate eigenvalues
    D = np.diag(1+rng.normal(size=n_features))**2
    if dtype is complex:
        Q = unitary_group.rvs(n_features, random_state=rng)
    else:
        Q = ortho_group.rvs(n_features, random_state=rng)
    Sigma = Q@D@Q.conj().T
    
    if unit_det:
        Sigma = Sigma/(np.real(det(Sigma))**(1/n_features))
    
    return Sigma.astype(dtype)



def sample_complex_gaussian(
    n_samples, location, shape, random_state=None
):
    """Sample from circular complex multivariate Gaussian distribution 

    Parameters
    ----------
    n_samples : int
        number of samples
    location : array-like of shape (n_features,)
        location (mean) of the distribution
    shape : array-like of shape (n_features, n_features)
        covariance of the distribution
    random_state : None or a numpy random data generator
        for reproducibility matters, one can provide a random generator
    """
    location_real = arraytoreal(location)
    shape_real = covariancetoreal(shape)
    return arraytocomplex(
        multivariate_normal.rvs(
            mean=location_real,
            cov=shape_real,
            size=n_samples,
            random_state=random_state
        )
    )



def arraytoreal(a):
    """Returns a real equivalent of input complex array used in various taks.
    Parameters
    ----------
    a : array-like of shape (n_samples, n_features)
        Input array.
    Returns
    -------
    array-like of shape (n_samples, 2*n_features)
        Real equivalent array.
    Raises
    ------
    AttributeError
        when input array format is not of dimension 1 or 2.
    """
    if np.iscomplexobj(a):
        if a.ndim == 1:
            return np.concatenate([np.real(a), np.imag(a)])
        elif a.ndim == 2:
            return np.hstack([np.real(a), np.imag(a)])
        else:
            raise AttributeError("Input array format not supported.")
    else:
        logging.debug("Input array is not complex, returning input")
        return a


def arraytocomplex(a):
    """Returns complex array from real input array.
    Parameters
    ----------
    a : array-like of shape (n_samples, 2*n_features)
        Input array.
    Returns
    -------
    array-like of shape (n_samples, 2*n_features)
        Real equivalent array.
    Raises
    ------
    AttributeError
        when input array format is not of dimension 1 or 2.
    """
    if not np.iscomplexobj(a):
        if a.ndim == 1:
            p = int(len(a)/2)
            return a[:p] + 1j*a[p:]
        elif a.ndim == 2:
            p = int(a.shape[1]/2)
            return np.vstack(a[:, :p] + 1j*a[:, p:])
        else:
            raise AttributeError("Input array format not supported")
    else:
        return a

def covariancetoreal(a):
    """Return real equivalent of complex matrix input.
    Parameters
    ----------
    a : array-like of shape (n_features, n_features)
        Input array.
    Returns
    -------
    array-like of shape (2*n_features, 2*n_features)
        Real equivalent of input array.
    Raises
    ------
    AttributeError
        when input array is not a covariance matrix.
    """

    if np.iscomplexobj(a):
        if iscovariance(a):
            real_matrix = .5 * np.block([[np.real(a), -np.imag(a)],
                                        [np.imag(a), np.real(a)]])
            return real_matrix
        else:
            raise AttributeError("Input array is not a covariance.")
    else:
        logging.debug("Input array is not complex, returning input.")
        return a


def covariancetocomplex(a):
    """Return complex matrix from its real equivalent in input.
    Input can be any transform of a matrix obtained thanks to function
    covariancetoreal or any square amtrix whose shape is an even number.
    Parameters
    ----------
    a : array-like of shape (2*n_features, 2*n_features)
        Input array, real equivalent of a complex square matrix.
    Returns
    -------
    array-like of shape (n_features, n_features)
        Real equivalent of input array.
    Raises
    ------
    AttributeError
        when input array format is not of dimension 2 or shape is not even.
    """

    if not np.iscomplexobj(a):
        if iscovariance(a) and len(a) % 2 == 0:
            p = int(len(a)/2)
            complex_matrix = 2 * a[:p, :p] + 2j*a[p:, :p]
            return complex_matrix
        else:
            raise AttributeError("Input array format not supported.")

    else:
        logging.debug("Input is already a complex array, returning input.")
        return a


def iscovariance(a):
    """Check if Input array correspond to a square matrix.
    TODO: do more than square matrix.
    Parameters
    ----------
    a : array-like
        Input array to check.
    Returns
    -------
    bool
        Return True if the input array is a square matrix.
    """

    return (a.ndim == 2) and (a.shape[0] == a.shape[1])


def matprint(mat, fmt="g"):
    """ Pretty print a matrix in Python 3 with numpy.
    Source: https://gist.github.com/lbn/836313e283f5d47d2e4e
    """

    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col])
                 for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


def compute_cor(statistic_H0: np.ndarray,
                statistic_H1: np.ndarray,
                n_points: int = 30) -> tuple:
    # Sorting H0 array to obtain threshold values
    threshold_values = np.sort(statistic_H0)[::-1]
    idx = np.round(np.linspace(0, len(statistic_H0) - 1, n_points)).astype(int)
    threshold_values = threshold_values[idx]

    # Measuring Pfa and Pd for each value of threshold
    P_fa, P_d = np.zeros((n_points,)), np.zeros((n_points,))
    for i, thresh_value in enumerate(threshold_values):
        P_fa[i] = np.sum(statistic_H0 >= thresh_value)/len(statistic_H0)
        P_d[i] = np.sum(statistic_H1 >= thresh_value)/len(statistic_H1)

    return P_fa, P_d
