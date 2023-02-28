import os, pickle
import numpy as np
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import trange
import argparse
from utility import sqrt_int

from wavelet_functions import decompose_image_wavelet
from change_detection import (
    covariance_equality_glrt_gaussian_statistic,
    scale_and_shape_equality_robust_statistic,
    scale_and_shape_equality_robust_statistic_kron,
    scale_and_shape_equality_robust_statistic_sgd,
    scale_and_shape_equality_robust_statistic_sgd_kron,
)
from multivariate_images_tools import sliding_windows_treatment_image_time_series_parallel

def compute_repeat_statistics(X, args):
    list_statistics, list_args, n_repeats = args
    return [ statistic(np.tile(X, n_repeats), arg) 
              for statistic, arg in 
              zip(list_statistics, list_args) ]


def parse_algorithms(detectors, a, b):
    
    list_names_possible = [
        'gaussian_glrt', 'scaled_gaussian_glrt',
        'scaled_gaussian_kron_glrt', 'scaled_gaussian_sgd',
        'scaled_gaussian_kron_glrt'
    ]
    
    list_detectors = []
    list_names = []
    list_args = []

    for statistic in detectors:
        if statistic == 'gaussian_glrt':
            list_detectors.append(covariance_equality_glrt_gaussian_statistic)
            list_names.append('Gaussian GLRT')
            list_args.append('log')
        elif statistic == 'scaled_gaussian_glrt':
            list_detectors.append(scale_and_shape_equality_robust_statistic)
            list_names.append('Scaled Gaussian GLRT')
            list_args.append((1e-4, 10, 'log'))
        elif statistic == 'scaled_gaussian_kron_glrt':
            list_detectors.append(scale_and_shape_equality_robust_statistic_kron)
            list_names.append('Scaled Gaussian Kronecker GLRT')
            list_args.append((a,b))
        elif statistic == 'scaled_gaussian_sgd':
            list_detectors.append(scale_and_shape_equality_robust_statistic_sgd)
            list_names.append('Scaled Gaussian SGD')
            list_args.append('Fixed-point')
        elif statistic == 'scaled_gaussian_kron_sgd':
            list_detectors.append(scale_and_shape_equality_robust_statistic_sgd_kron)
            list_names.append('Scaled Gaussian Kronecker SGD')
            list_args.append((a, b))
        else:
            raise AssertionError(
                f'{statistic} algorithm is not recognised. Choices are: {list_names_possible}'
            )
    
    return list_detectors, list_args, list_names

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Change detection on UAVSAR images')
    parser.add_argument('results_dir', metavar='r', type=str,
                        help='Directory where to store the results')
    parser.add_argument('scene', metavar='s', type=int, default=1,
                        help='Scene to compute the change detection')
    parser.add_argument('n_repeats', type=int, default=1,
                        help='number of times we repeat time series')
    parser.add_argument('-c','--crop_indexes', nargs='+', 
                        help='Cropping indexes of the image', 
                        required=False, default=None) 
    parser.add_argument('-d','--detectors', nargs='+', 
                        help='name of algorithms to compute. Choice between gaussian_glrt, scaled_gaussian_glrt,'+\
                             'scaled_gaussian_kron_glrt, scaled_gaussian_sgd, scaled_gaussian_kron_sgd', 
                        required=True)
    args = parser.parse_args()

    if args.crop_indexes is not None and len(args.crop_indexes)!=4:
        raise AssertionError(f'{args.crop_indexes} is not to the right format!')
    if args.n_repeats < 1:
        raise AssertionError(f'n_repeats must be greater or equal to one !')

    # -----------------------------------------------------------------------------
    # Parameters
    # -----------------------------------------------------------------------------
    data_path = "./data/UAVSAR"
    scene = args.scene
    image_path = os.path.join(data_path, f"Scene_{scene}.npy")
    crop_indexes = args.crop_indexes
    if crop_indexes is not None:
        crop_indexes = [int(x) for x in args.crop_indexes]

    # Data dimensionality parameters
    n_repeats = args.n_repeats # number of times we repeat the time series
    a = 4
    b = 3

    # Wavelet parameters : R*L must be equal to a
    R = 2
    L = 2
    d_1 = 10
    d_2 = 10

    # Sliding windows parametera
    mask = (7, 7)

    # -----------------------------------------------------------------------------
    # Detectors
    # -----------------------------------------------------------------------------
    list_detectors, list_args, list_names = parse_algorithms(args.detectors, a, b)

    # -----------------------------------------------------------------------------
    # Results directory and log management
    # -----------------------------------------------------------------------------
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    # Redirecting tqdm to file
    f_tqdm = open(os.path.join(args.results_dir, f'progress_{args.detectors}.log'), 'w')

    # -----------------------------------------------------------------------------
    # Data reading + Wavelet decomposition
    # -----------------------------------------------------------------------------
    print(f'Reading data from {image_path}')
    sits_data = np.load(image_path)
    if crop_indexes is not None:
        sits_data = sits_data[
            crop_indexes[0]:crop_indexes[1], 
            crop_indexes[2]:crop_indexes[3]
        ]


    print("Doing wavelet decomposition")
    center_frequency = 1.26e+9 # GHz, for L Band
    bandwith = 80.0 * 10**6 # Hz
    range_resolution = 1.66551366 # m, for 1x1 slc data
    azimuth_resolution = 0.6 # m, for 1x1 slc data
    number_pixels_azimuth, number_pixels_range, p, T = sits_data.shape
    range_vec = np.linspace(-0.5,0.5,number_pixels_range) * range_resolution * number_pixels_range
    azimuth_vec = np.linspace(-0.5,0.5,number_pixels_azimuth) * azimuth_resolution * number_pixels_azimuth
    Y, X = np.meshgrid(range_vec,azimuth_vec)

    image = np.zeros(
        (number_pixels_azimuth, number_pixels_range, p*R*L, T), 
        dtype=complex
    )
    for t in range(T):
        for i_p in range(p):
            image_temp = decompose_image_wavelet(
                sits_data[:,:,i_p,t], bandwith, range_resolution, 
                azimuth_resolution, center_frequency, R, L, d_1, d_2
            )
            image[:,:,i_p*R*L:(i_p+1)*R*L, t] = image_temp
    sits_data = image
    n_rows, n_cols, n_features, n_times  = sits_data.shape
    print('Done')


    # -----------------------------------------------------------------------------
    # Performing detection
    # -----------------------------------------------------------------------------
    print('Performing detection')
    n_threads_rows, n_threads_columns = sqrt_int(os.cpu_count())
    results = sliding_windows_treatment_image_time_series_parallel(
        sits_data, np.ones(mask), compute_repeat_statistics, (list_detectors, list_args, n_repeats),
        multi=True, number_of_threads_rows=n_threads_rows, number_of_threads_columns=n_threads_columns,
        tqdm_out=f_tqdm
    )
    f_tqdm.close()
    print('Done.')

    # -----------------------------------------------------------------------------
    # Saving data
    # -----------------------------------------------------------------------------
    metadata_path = os.path.join(args.results_dir, 'metadata.txt')
    if not os.path.exists(metadata_path):
        with open(os.path.join(args.results_dir, 'metadata.txt'), 'w') as f:
            f.write(f'scene: {scene}\n')
            f.write(f'crop_indexes: {crop_indexes}\n')
            f.write(f'n_repeats: {n_repeats}\n')
            f.write(f'a: {a}\n')
            f.write(f'b: {b}\n')
            f.write(f'mask: {mask}\n')


    for i, name in enumerate(args.detectors):
        artifact_path = os.path.join(args.results_dir, f'artifact_{name}.pkl')
        print(f'Saving artifact to {artifact_path}')
        tosave = {
            'scene': scene,
            'crop_indexes': crop_indexes,
            'n_repeats': n_repeats,
            'a': a, 'b':b,
            'list_names': list_names,
            'list_args': list_args,
            'results': np.array(results[:,:,i])
        }
        with open(artifact_path, 'wb') as f:
            pickle.dump(tosave, f)
    print('Done.')

