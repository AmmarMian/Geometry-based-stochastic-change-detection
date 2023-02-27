# Small script to test parallel processing of an multivariate image
# Through a sliding windows approach

import numpy as np
from joblib import Parallel, delayed
from change_detection import covariance_equality_glrt_gaussian_statistic
from tqdm import trange
from multivariate_images_tools import sliding_windows_treatment_image_time_series_parallel
from compute_CD_realdata import vectorize_spatial

def treatment(X, args):
    return [covariance_equality_glrt_gaussian_statistic(X, 'log')] 

if __name__ == "__main__":
    
    image = np.random.randn(1000, 500, 12, 5) + 1j* np.random.randn(1000, 500, 12, 5)

    # multivariate_image_tools
    print("Doing multivariate_image_tools way")
    mask = np.ones((7,7))
    result = sliding_windows_treatment_image_time_series_parallel(image, mask, treatment, None,
                multi=True, number_of_threads_rows=8, number_of_threads_columns = 4)

    # joblib naive
    print("Doing joblib way")
    view = np.lib.stride_tricks.sliding_window_view(image, (7,7), axis=[0,1])
    result = Parallel(n_jobs=32)(
        delayed(treatment)(
            np.moveaxis(vectorize_spatial(view[line,column]), 2, 1), None
        ) 
        for line in trange(view.shape[0])
        for column in range(view.shape[1])
    )
