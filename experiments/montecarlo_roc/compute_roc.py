# ========================================
# FileName: change_detection_roc.py
# Date: 21 mars 2023 - 11:10
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: ROC for a simulated change
# detection scenario.
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
    compute_cor,
    sample_complex_gaussian
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Monte-carlo Change Detection ROC simulation')
    parser.add_argument('config_file', type=str,
                        help="Experiment parameters file: python file defining:"
                        " a, b, A_H0, B_H0, A_H1, B_H1, n_trials, n_points_roc,"
                        " n_batches_list, statistics, batch_no_change, n_samples")
    parser.add_argument('results_dir', metavar='r', type=str,
                        help='Directory where to store the results')
    args = parser.parse_args()


    # Importing variables from config_file
    sys.path.append(os.path.dirname(args.config_file))
    config = import_module(Path(args.config_file).stem)
    Sigma_H0, Sigma_H1 = np.kron(config.A_H0, config.B_H0), np.kron(config.A_H1, config.B_H1)

    # Redirecting tqdm to file
    f_tqdm = open(os.path.join(args.results_dir, 'progress.log'), 'w')

    # Definition of a single MonteCarlo Trial
    def one_trial(trial_no):
        rng = np.random.default_rng(trial_no)

        # To save time, we generate data with the max number of batches
        # then just take the increasing needed  number of batches
        n_batches = max(config.n_batches_list)
        # Generating data H0
        X_H0_global = np.zeros((config.a*config.b, config.n_samples, n_batches), dtype=complex)
        tau_0 = rng.gamma(config.nu_0, size=config.n_samples)
        for batch_no in range(n_batches):
             X_H0_global[..., batch_no] = sample_complex_gaussian(
                        config.n_samples, np.zeros((config.a*config.b,), dtype=complex),
                        Sigma_H0, random_state=rng).T * np.sqrt(tau_0)

        # Generating no change data part of H1 scenario
        tau_1 = rng.gamma(config.nu_1, size=config.n_samples)
        X_H1_global = np.zeros((config.a*config.b, config.n_samples,
                                config.batch_no_change(n_batches)),
                               dtype=complex)
        for batch_no in range(config.batch_no_change(n_batches)):
            X_H1_global[..., batch_no] = sample_complex_gaussian(
                        config.n_samples, np.zeros((config.a*config.b,), dtype=complex),
                        Sigma_H0, random_state=rng).T * np.sqrt(tau_0)

        results = {'trial_no': trial_no}
        for n_batches in config.n_batches_list:
            # Organising data from the corresponding number of batches
            X_H0 = X_H0_global[..., :n_batches]
            X_H1 = np.zeros((config.a*config.b, config.n_samples, n_batches), dtype=complex)
            X_H1[..., :config.batch_no_change(n_batches)] = X_H1_global[..., :config.batch_no_change(n_batches)]
            for batch_no in range(config.batch_no_change(n_batches), n_batches):
                X_H1[..., batch_no] = sample_complex_gaussian(
                            config.n_samples, np.zeros((config.a*config.b,), dtype=complex),
                            Sigma_H1, random_state=rng).T * np.sqrt(tau_1)

            # Computing statistics values on data
            results[n_batches] = {}
            for X, scenario in zip([X_H0, X_H1], ['H0', 'H1']):
                results[n_batches][scenario] = {}
                for statistic_name in config.statistics:
                    results[n_batches][scenario][statistic_name] = \
                            config.statistics[statistic_name]['function'](
                                X, config.statistics[statistic_name]['args']
                            )
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
    markers = ['o', 'x', '+', '□', '◇', '⊗', '⌀', '⏹']
    results_tosave = {}
    for n_batches in config.n_batches_list:
        fig = plotille.Figure()
        fig.height = 15
        fig.width = 50
        fig.x_label = 'Pfa'
        fig.y_label = 'Pd'
        fig.set_x_limits(min_=0, max_=1.05)
        fig.set_y_limits(min_=0, max_=1.05)
        results_tosave[n_batches] = {}
        for i, statistic_name in enumerate(config.statistics):
            statistic_values_H0 = [results[trial_no][n_batches]['H0'][statistic_name]
                                   for trial_no in range(config.n_trials)]
            statistic_values_H1 = [results[trial_no][n_batches]['H1'][statistic_name]
                                   for trial_no in range(config.n_trials)]

            pfa, pd = compute_cor(
                    np.array(statistic_values_H0),
                    np.array(statistic_values_H1),
                    config.n_points_roc)
            results_tosave[n_batches][statistic_name] = {'pfa': pfa, 'pd': pd}
            fig.plot(pfa, pd, label=statistic_name, marker=markers[i])

        rprint(f"[bold red]Simulation with n_batches={n_batches}")
        print('\n')
        print(fig.show(legend=True))
        print('\n\n\n')

    # Plot comparing GLRT with SGD
    for statistic_name_glrt, statistic_name_sgd in zip(
        ['Scaled Gaussian GLRT', 'Scaled Gaussian Kronecker GLRT'],
        ['Scaled Gaussian SGD', 'Scaled Gaussian Kronecker SGD']):
        fig = plotille.Figure()
        fig.height = 15
        fig.width = 50
        fig.x_label = 'Pfa'
        fig.y_label = 'Pd'
        fig.set_x_limits(min_=0, max_=1.05)
        fig.set_y_limits(min_=0, max_=1.05)

        statistic_values_H0 = [results[trial_no][config.n_batches_list[-1]]['H0'][statistic_name_glrt]
                               for trial_no in range(config.n_trials)]
        statistic_values_H1 = [results[trial_no][config.n_batches_list[-1]]['H1'][statistic_name_glrt]
                               for trial_no in range(config.n_trials)]
        pfa, pd = compute_cor(
                np.array(statistic_values_H0),
                np.array(statistic_values_H1),
                config.n_points_roc)
        fig.plot(
            pfa, pd,
            label=f'{statistic_name_glrt}: n_batches={config.n_batches_list[-1]}',
            marker='⎈'
        )
        for i, n_batches in enumerate(config.n_batches_list):
            statistic_values_H0 = [results[trial_no][n_batches]['H0'][statistic_name_sgd]
                                   for trial_no in range(config.n_trials)]
            statistic_values_H1 = [results[trial_no][n_batches]['H1'][statistic_name_sgd]
                                   for trial_no in range(config.n_trials)]

            pfa, pd = compute_cor(
                    np.array(statistic_values_H0),
                    np.array(statistic_values_H1),
                    config.n_points_roc)
            fig.plot(
                pfa, pd,
                label=f'{statistic_name_sgd}: n_batches={n_batches}',
                marker=markers[i]
            )
        rprint(f"[bold red]Results for {statistic_name_glrt}:")
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
    }
    with open(artifact_path, 'wb') as f:
        pickle.dump(tosave, f)
    print('Done.')

