import os, pickle
import numpy as np
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import trange
import argparse

from wavelet_functions import decompose_image_wavelet
from change_detection import (
    covariance_equality_glrt_gaussian_statistic,
    scale_and_shape_equality_robust_statistic,
    scale_and_shape_equality_robust_statistic_kron,
    scale_and_shape_equality_robust_statistic_sgd,
    scale_and_shape_equality_robust_statistic_sgd_kron,
)


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

    # Creating a view of the data in the form of sliding windows
    view = sliding_window_view(sits_data, mask, (0, 1))


    # -----------------------------------------------------------------------------
    # Performing detection
    # -----------------------------------------------------------------------------
    print('Performing detection')
    memap_filepath = os.path.join(args.results_dir, f'result__{args.detectors}.dat')
    # Allocating memory on disk and mapping it to array fro storing the results
    if os.path.exists(memap_filepath):
        # In case the computation has been halted, we don't start from scratch
        results = np.memmap(
            memap_filepath,
            dtype=np.float32, mode="r+",
            shape=view.shape[:2]+(len(list_detectors),)
        )
    else:
        results = np.memmap(
            memap_filepath,
            dtype=np.float32, mode="w+",
            shape=view.shape[:2]+(len(list_detectors),)
        )
        results[:] = np.nan
        results.flush()


    # Looking for first nan to find where we start again the computation 
    status = np.where(np.isnan(results))
    start_line, start_column = status[0][0], status[1][0]

    # We only want to start to column fro the current line
    def range_column(line):
        if line == start_line:
            return range(start_column, view.shape[1])
        else:
            return range(view.shape[1])

    Parallel(n_jobs=-1)(
        delayed(compute_several_statistics_tomemmap)(
            # To repeat the data in case we don't have enough samples
            np.tile(
                # To have an array of shape (p, N, T)
                np.moveaxis(vectorize_spatial(view[line,column]), 2, 1), n_repeats
            ), list_detectors, list_args, results, line, column
        )
        for line in trange(start_line, view.shape[0], file=f_tqdm)
        for column in range_column(line)
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

    # Cleanup
    try:
        os.remove(memap_filepath)
    except:  # noqa
        print('Could not clean-up automatically.')
