import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import numpy.linalg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Script for plotting results of "
            "Change Detection using robust statistics"
    )
    parser.add_argument('results_dir',
                        help="Directory in which results "
                        " from compute_cd_realdata.py are stored")
    parser.add_argument('--usetex', required=False, default=False,
                        help="Whether to use LaTeX for rendering the figures")
    parser.add_argument('--savefig', required=False, default=False,
                        help="Wheter to save figures in the results_dir")
    parser.add_argument('--data_dir', required=False, default='./data/UAVSAR',
                        help="Directory where data is stored to plot "
                        "time series of images")
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

    # Parsing file system for results_direcotry
    list_artifacts_steps = glob.glob(
        os.path.join(args.results_dir, "**/artifact*.pkl")
    )
    list_names = []
    results = []
    for artifact in list_artifacts_steps:
        with open(artifact, 'rb') as f:
            data = pickle.load(f)
            list_names += data['list_names']
            results.append(data['results'])
            scene = data['scene']
            crop_indexes = data['crop_indexes']

    # Stacking results in one numpy array
    results = np.dstack(results)

    # ---------------------------------------------------- 
    # Plotting
    # ---------------------------------------------------- 

    # SAR images
    # with open(os.path.join(args.data_dir, f'Scene_{scene}.npy'), 'rb') as f:
        # sits = np.load(f)
    # for t in range(sits.shape[-1]):
        # plt.figure(figsize=(16, 10))
        # span = 20*np.log10(
            # np.sum(np.abs(sits[...,t])**2, axis=2)
        # )
        # plt.imshow(span, cmap='gray', aspect='auto')
        # plt.colorbar()
        # plt.title(f'Image at t={t}')

    # Results of detectors
    for i, name in enumerate(list_names):
        plt.figure(figsize=(16, 10))
        plt.imshow(results[...,i], cmap='jet', aspect='auto')
        plt.colorbar()
        plt.title(f'Result of test statistic: {name}')

    plt.show()


