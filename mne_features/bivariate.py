# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


from functools import partial
from math import sqrt
import numpy as np
from .mock_numba import nb


def get_bivariate_funcs(sfreq):
    """ Returns a dictionary of bivariate feature functions. For each feature
    function, the corresponding key in the dictionary is an alias for the
    function.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    Returns
    -------
    bivariate_funcs : dict of feature functions
    """
    bivariate_funcs = dict()
    bivariate_funcs['max_cross_corr'] = partial(compute_max_cross_correlation,
                                                sfreq)
    return bivariate_funcs


@nb.jit(nb.float64(nb.float64[:], nb.float64[:], nb.int64))
def _cross_correlation(x, y, tau):
    """ Computes the cross-correlation between two univariate time series
    x and y, with a delay tau. The function assumes that the time series x and
    y have the same length. The parameter tau must be strictly less than the
    length of the time series.

    Parameters
    ----------
    x : array-like, shape (n_times,)

    y : array-like, shape (n_times,)

    tau : int
        Delay (number of samples)

    Returns
    -------
    float
    """
    if tau < 0:
        _tau = -tau
    else:
        _tau = tau
    n_times = x.shape[0]
    x_m = 0
    y_m = 0
    for j in range(n_times):
        x_m += x[j]
        y_m += y[j]
    x_m /= n_times
    y_m /= n_times
    x_v = 0
    y_v = 0
    for j in range(n_times):
        x_v += (x[j] - x_m) * (x[j] - x_m)
        y_v += (y[j] - y_m) * (y[j] - y_m)
    x_v /= (n_times - 1)
    y_v /= (n_times - 1)
    x_v = sqrt(x_v)
    y_v = sqrt(y_v)
    cc = 0
    for j in range(0, n_times - _tau):
        cc += ((x[j + _tau] - x_m) / x_v) * ((y[j] - y_m) / y_v)
    cc /= (n_times - _tau)
    return cc


@nb.jit(nb.float64(nb.float64[:], nb.float64[:], nb.int64[:]))
def _max_cross_corr(x, y, taus):
    """ Computes the maximum cross-correlation between two univariate time
    series using a range of possible time delays.

    Parameters
    ----------
    x : array-like, shape (n_times,)

    y : array-like, shape (n_times,)

    taus : array-like, shape (n_tau,)

    Returns
    -------
    float
    """
    n_tau = taus.shape[0]
    res = np.empty((n_tau,))
    for j in range(n_tau):
        res[j] = abs(_cross_correlation(x, y, taus[j]))
    return np.max(res)


def compute_max_cross_correlation(s_freq, data):
    """ Computes max cross-correlation for pairs of channels. """

    n_channels = data.shape[0]
    n_tau = int(0.5 * s_freq)
    taus = np.arange(-n_tau, n_tau, dtype=np.int64)
    n_coefs = n_channels * (n_channels + 1) // 2
    max_cc = np.zeros((n_coefs,))
    idx0, idx1 = np.triu_indices(n_channels)
    for j in range(n_coefs):
        max_cc[j] = _max_cross_corr(data[idx0[j], :], data[idx1[j], :], taus)
    return max_cc.ravel()
