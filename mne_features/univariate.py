# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


import numpy as np
from scipy import stats


def get_univariate_funcs():
    """ Returns a dictionary of univariate feature functions. For each feature
    function, the corresponding key in the dictionary is an alias for the
    function.

    Returns
    -------
    univariate_funcs : dict
    """
    univariate_funcs = dict()
    univariate_funcs['mean'] = compute_mean
    univariate_funcs['variance'] = compute_variance
    univariate_funcs['ptp_amplitude'] = compute_ptp
    univariate_funcs['skewness'] = compute_skewness
    univariate_funcs['kurtosis'] = compute_kurtosis
    return univariate_funcs


def compute_mean(data):
    """ Computes the mean of the data. """

    ndim = data.ndim
    return np.mean(data, axis=ndim - 1)


def compute_variance(data):
    """ Computes the variance of the data. """

    ndim = data.ndim
    return np.var(data, axis=ndim - 1, ddof=1)


def compute_ptp(data):
    """ Compute peak-to-peak amplitude. """
    return np.ptp(data, axis=-1) ** 2


def compute_skewness(data):
    """ Computes the skewness of the data. """

    ndim = data.ndim
    return stats.skew(data, axis=ndim - 1)


def compute_kurtosis(data):
    """ Computes the kurtosis of the data. """

    ndim = data.ndim
    return stats.kurtosis(data, axis=ndim - 1, fisher=False)
