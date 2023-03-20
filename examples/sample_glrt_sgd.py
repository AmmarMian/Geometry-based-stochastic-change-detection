# ========================================
# FileName: sample_glrt_sgd.py
# Date: 16 mars 2023 - 11:50
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: Monte-carlo performance estimation
# of detectors in form of ROC curves
# =========================================
import sys
import os
sys.path.append(os.path.join(os.path.dirname(sys.argv[0]), '../src'))

from tqdm import trange, tqdm
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from functions import (
    generate_covariance_toeplitz,
    multivariate_complex_normal_samples
)
from change_detection import (
    covariance_equality_glrt_gaussian_statistic,
    scale_and_shape_equality_robust_statistic,
    scale_and_shape_equality_robust_statistic_kron,
    scale_and_shape_equality_robust_statistic_sgd,
    scale_and_shape_equality_robust_statistic_sgd_kron   
)


def sample_statistics(
        p, N, n_batches, cov,
        list_statistics, list_args, seed):

    np.random.seed(seed)
    X = np.zeros((p, N, n_batches), dtype=complex)
    for n in range(n_batches):
        X[:, :, n] = multivariate_complex_normal_samples(
                np.zeros((p,), dtype=complex),
                cov, N
            )

    lbda = []
    for statistic, args in zip(list_statistics, list_args):
        lbda.append(
            statistic(X, args)
        )

    return lbda


if __name__ == '__main__':

    # Simulations parameters
    a = 2
    b = 2
    p = a*b
    N = 9
    n_batches_list = np.unique(np.logspace(1, 3, 2, base=p, dtype=int))
    n_trials = 10

    A = generate_covariance_toeplitz(0.1+0.3j, a)
    B = generate_covariance_toeplitz(0.7+0.6j, b)
    A = A/(np.linalg.det(A)**(1/a))
    B = 10*B/(np.linalg.det(B)**(1/b))
    # For numerical stability
    A = (A + A.conj().T)/2
    B = (B + B.conj().T)/2
    cov = np.kron(A, B)

    # Definition of statistics used
    list_statistics_classic = [
        scale_and_shape_equality_robust_statistic,
        scale_and_shape_equality_robust_statistic_kron,
    ]
    list_args_classic = [
        (1e-4, 10, 'log'),
        (a, b),
    ]
    list_names_classic = [
        "Scaled Gaussian GLRT",
        "Scaled Gaussian Kronecker GLRT",
    ]

    list_statistics_sgd = [
        scale_and_shape_equality_robust_statistic_sgd,
        scale_and_shape_equality_robust_statistic_sgd_kron,
    ]
    list_args_sgd = [
        'Fixed-point',
        (a, b),
    ]
    list_names_sgd = [
        "Scaled Gaussian GLRT SGD",
        "Scaled Gaussian Kronecker GLRT SGD",
    ]

    # Doing parallel sampling of classic algorithms
    print("Doing Monte-Carlo for True GLRT")
    LBDA_classic = np.array(
        Parallel(n_jobs=-1)(
            delayed(sample_statistics)(
                    p, N, n_batches_list[-1], cov,
                    list_statistics_classic, list_args_classic,
                    trial
            )
            for trial in trange(n_trials)
        )
    )

    # Doing parallel sampling for SGD algorithms
    print("Doing Monte-carlo for SGD algorithms")
    LBDA_SGD = np.array(
        Parallel(n_jobs=-1)(
            delayed(sample_statistics)(
                    p, N, n_batches, cov,
                    list_statistics_sgd, list_args_sgd,
                    trial
            )
            for n_batches in tqdm(n_batches_list)
            for trial in range(n_trials)
        )
    ).reshape((n_trials, len(n_batches_list), len(list_statistics_sgd)))

    # Plotting
    fig, slider, ax_slider = {}, {}, {}
    for i in range(len(list_statistics_classic)):
        fig[i] = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        fig[i][1].hist(LBDA_classic[:, i], label='True GLRT')
        fig[i][1].hist(LBDA_SGD[:, 0, i], label=f'SGD {n_batches_list[0]} batches')
        plt.title(f'{list_names_classic[i]} - {n_batches_list[0]}')

        # Make a horizontal slider to control the number of batches
        ax_slider[i] = fig[i][0].add_axes([0.25, 0.1, 0.65, 0.03])
        slider[i] = Slider(
            ax=ax_slider[i],
            label=r'index $n_{\mathrm{batches}}$',
            valmin=0,
            valmax=len(n_batches_list)-1,
            valinit=0,
            valstep=1,
        )

        def update(val, figtemp):
            figtemp[1].clear()
            figtemp[1].hist(LBDA_classic[:, i], label='True GLRT')
            figtemp[1].hist(LBDA_SGD[:, val, i], label=f'SGD {n_batches_list[val]} batches')
            figtemp[0].suptitle(f'{list_names_classic[i]} - {n_batches_list[val]}')

        slider[i].on_changed(lambda x: update(x, fig[i]))

        # plt.title(list_names_classic[i])
        plt.show()
