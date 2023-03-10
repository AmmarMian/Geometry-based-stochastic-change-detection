import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from pprint import pprint
import seaborn as sns
from change_detection import Computing_COR_ChangeDetection

sns.set_theme(
    style='darkgrid',
    font='serif'
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Script for plotting results of "
            "Change Detection using robust statistics. Version "
            "with Monte-carlo simulations."
    )
    parser.add_argument('results_dir',
                        help="Directory in which results "
                        " from compute_cd_realdata.py are stored")
    parser.add_argument('--usetex', required=False, default=False,
                        action='store_true',
                        help="Whether to use LaTeX for rendering the figures")
    parser.add_argument('--savefig', required=False, default=False,
                        help="Wheter to save figures in the results_dir")
    parser.add_argument('--n_points', required=False, default=30,
                        help="Number of points for the ROC plot")
    args = parser.parse_args()

    # Activating LaTeX or not
    if args.usetex:
        plt.rcParams.update({"text.usetex": True})

    # Checking if results_dir exists
    if not os.path.exists(args.results_dir):
        raise FileNotFoundError(
                f'Results directory {args.results_dir} does not exists!'
        )
        sys.exit(0)

    # Loading artifact into memory
    print(f'Reading artfact data from directory {args.results_dir}')
    with open(os.path.join(args.results_dir, "artifact.pkl"), 'rb') as f:
        simulation_data = pickle.load(f)

    # ----------------------------------------------------
    # Plotting
    # ----------------------------------------------------

    # Get list of all detectors used from pickled dict
    list_names = [
        x for x in simulation_data.keys()
        if isinstance(simulation_data[x], dict)
    ]
    simulation_info = dict(
            filter(lambda pair: pair[0] not in list_names,
                   simulation_data.items())
    )
    pprint(simulation_info)

    # ROC Results of detectors
    plt.figure(figsize=(8, 6))
    for name in list_names:
        Pd, Pfa = Computing_COR_ChangeDetection(
            np.array(simulation_data[name]['result_H0']),
            np.array(simulation_data[name]['result_H1']),
            args.n_points
        )
        plt.plot(Pfa, Pd, label=name)
    plt.legend()
    plt.xlabel(r'$P_{\mathrm{FA}}$')
    plt.ylabel(r'$P_{\mathrm{D}}$')

    # Histogram of statistics under H0 and H1
    for name in list_names:
        fig, ax = plt.subplots(nrows=2, figsize=(9, 8))

        # H0
        sns.histplot(
            np.array(simulation_data[name]['result_H0']),
            ax=ax[0], common_norm=True
        )
        ax[0].set_title('Distribution under $H_0$')
        ax[0].set_ylabel('Count')

        # H1
        sns.histplot(
            np.array(simulation_data[name]['result_H1']),
            ax=ax[1], common_norm=True 
        )
        ax[1].set_title('Distribution under $H_1$')
        ax[1].set_xlabel('$\lambda$')
        ax[1].set_ylabel('Count')
        fig.suptitle(f'Distribution of {name}')
    plt.show()
