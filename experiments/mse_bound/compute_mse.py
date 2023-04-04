# ========================================
# FileName: compute_mse.py
# Date: 21 mars 2023 - 11:10
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: Compute the MSE of the estimators
# =========================================

import argparse
import numpy as np
from joblib import Parallel, delayed
from tqdm import trange, tqdm
from rich import print as rprint
import plotille
from importlib import import_module
from pathlib import Path
import pickle
import sys
import os
sys.path.append(os.path.join(os.path.dirname(sys.argv[0]), '../../src'))


from utility import (
    sample_complex_gaussian
)


def parse_results(results: list, type_algo: str, config: any) -> dict:
    """Parse results list to compute mean along trials for each estimator.
    type_algo = 'offline' or 'online'"""
    results_dict = {}
    for estimator in results[0][config.n_batches_list[0]][type_algo]:
        results_dict[estimator] = {}
        for element in ['A', 'B', 'tau']:
            mse_element = np.mean(
                np.array([[results[trial_no][n_batches][type_algo][estimator][element]
                           for n_batches in config.n_batches_list]
                          for trial_no in range(config.n_trials)]),
                axis=0)
            results_dict[estimator][element] = mse_element
    return results_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Monte-carlo Change Detection ROC simulation')
    parser.add_argument('config_file', type=str,
                        help="Experiment parameters file: python file defining:"
                        " a, b, A, B, tau, n_trials, n_samples, n_batches_list"
                        " estimators_offline, estimators_online, mse_function_A"
                        " mse_function_B, mse_function_tau")
    parser.add_argument('results_dir', metavar='r', type=str,
                        help='Directory where to store the results')
    args = parser.parse_args()

    # Importing variables from config_file
    sys.path.append(os.path.dirname(args.config_file))
    config = import_module(Path(args.config_file).stem)
    Sigma = np.kron(config.A, config.B)

    # Redirecting tqdm to file
    f_tqdm = open(os.path.join(args.results_dir, 'progress.log'), 'w')

    # Definition of a single MonteCarlo Trial
    def one_trial(trial_no):
        """One Montecarlo Tiral for computing MSE of estimators
        online and offline."""
        rng = np.random.default_rng(trial_no)

        # To save time, we generate data with the max number of batches
        # then just take the increasing needed  number of batches
        n_batches = max(config.n_batches_list)
        # Generating data H0
        X_global = np.zeros((config.a*config.b, config.n_samples, n_batches),
                            dtype=complex)
        for batch_no in range(n_batches):
            X_global[..., batch_no] = sample_complex_gaussian(
                        config.n_samples,
                        np.zeros((config.a*config.b,), dtype=complex),
                        Sigma, random_state=rng).T * np.sqrt(config.tau.flatten())

        results = {'trial_no': trial_no}

        # Initialising estimates online
        A_est_online, B_est_online, tau_est_online = {}, {}, {}
        for estimator in config.estimators_online:
            A_est_online[estimator] = config.estimators_online[estimator]['init'][0]
            B_est_online[estimator] = config.estimators_online[estimator]['init'][1]
            tau_est_online[estimator] = config.estimators_online[estimator]['init'][2]
        n_batches_start = 0

        for n_batches in config.n_batches_list:
            results[n_batches] = {'offline': {}, 'online': {}}

            # Organising data from the corresponding number of batches
            X = X_global[..., :n_batches]

            # Computing estimate offline
            for estimator in config.estimators_offline:
                A_est, B_est, tau_est = config.estimators_offline[estimator](X)
                results[n_batches]['offline'][estimator] = {
                    'A': config.mse_function_A(A_est, config.A),
                    'B': config.mse_function_B(B_est, config.B),
                    'tau': config.mse_function_tau(tau_est, config.tau)
                }

            # Computing estimate online by doing stochastic gradient descent
            # iteration
            for estimator in config.estimators_online:
                # Doing as many iterations as needed to get to the current
                # number of batches
                for i_batch in range(n_batches_start, n_batches):
                    # Computing riemannian gradient
                    r_A, r_B, r_tau = config.estimators_online[estimator]['rgrad'](
                            X_global[..., i_batch].T, A_est_online[estimator],
                            B_est_online[estimator],
                            tau_est_online[estimator])
                    # Retraction on the manifold
                    A_est_online[estimator], B_est_online[estimator], tau_est_online[estimator] = \
                        config.estimators_online[estimator]['manifold'].retr(
                           (A_est_online[estimator], B_est_online[estimator], tau_est_online[estimator]),
                           [-(config.estimators_online[estimator]['lr']/(i_batch+1))*r_x
                            for r_x in [r_A, r_B, r_tau]]
                        )
                n_batches_start = n_batches

                results[n_batches]['online'][estimator] = {
                    'A': config.mse_function_A(A_est_online[estimator], config.A),
                    'B': config.mse_function_B(B_est_online[estimator], config.B),
                    'tau': config.mse_function_tau(tau_est_online[estimator], config.tau)
                }

        return results

    # Launching parallel processing
    rprint('[bold]Launching simulation with parameters:')
    print(f'a={config.a}, b={config.b}, n_samples={config.n_samples}, n_batches_list={config.n_batches_list}')
    results = Parallel(n_jobs=-1)(
            delayed(one_trial)(trial_no)
            for trial_no in trange(config.n_trials, file=f_tqdm)
        )
    f_tqdm.close()

    # Organising results for plots
    results_element_offline = parse_results(results, 'offline', config)
    results_element_online = parse_results(results, 'online', config)
    results_tosave = {
                'online': results_element_online,
                'offline': results_element_offline
            }

    markers = ['o', 'x', '+', '□', '◇', '⊗', '⌀', '⏹']
    ICRB_A = lambda n: (config.a**2-1)/(config.b*config.n_samples*n)
    ICRB_B = lambda n: (config.b**2-1)/(config.a*config.n_samples*n)
    ICRB_tau = lambda n: 1/(n*config.a*config.b)
    for element, icrb in zip(['A', 'B', 'tau'], [ICRB_A, ICRB_B, ICRB_tau]):
        fig = plotille.Figure()
        fig.height = 15
        fig.width = 50
        fig.x_label = 'log(Number of batches)'
        fig.y_label = f'log(MSE) {element}'

        for i, results_dict in enumerate(
                [results_element_offline, results_element_online]):
            for j, estimator in enumerate(results_dict):
                fig.plot(np.log10(config.n_batches_list),
                         np.log10(results_dict[estimator][element]),
                         label=estimator,
                         marker=markers[i*2+j])

        # Computing ICRB
        icrb_element = icrb(np.array(config.n_batches_list))
        fig.plot(np.log10(config.n_batches_list), np.log10(icrb_element),
                 label=f'ICRB {element}', marker='*')

        print('\n')
        print(fig.show(legend=True))
        print('\n\n\n')

    # Saving results to result directory
    rprint(f'[bold green]Saving results in {args.results_dir}')
    artifact_path = os.path.join(args.results_dir, 'artifact.pkl')
    print(f'Saving artifact to {artifact_path}')
    tosave = {
        'config_file': args.config_file,
        'results': results_tosave,
        'results_raw': results
    }
    with open(artifact_path, 'wb') as f:
        pickle.dump(tosave, f)
    print('Done.')

