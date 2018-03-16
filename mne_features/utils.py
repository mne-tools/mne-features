# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


""" Utility functions to be used with either univariate or bivariate feature
functions."""

from math import floor
from warnings import warn
import numpy as np

from .mock_numba import nb


@nb.jit()
def triu_idx(n):
    """ Utility function to generate an enumeration of the pairs of indices
    (i,j) corresponding to the upper triangular part of a (n, n) array.

    Parameters
    ----------
    n : int

    Returns
    -------
    generator
    """
    pos = -1
    for i in range(n):
        for j in range(i, n):
            pos += 1
            yield pos, i, j


def embed(x, d, tau):
    """ Utility function to compute the time-delay embedding of a univariate
    time series x.

    Parameters
    ----------
    x : ndarray, shape (n_times,)

    d : int
        Embedding dimension.
        The embedding dimension `d` should be greater than 2.

    tau : int
        Delay.
        The delay parameter `tau` should be less or equal than
        `floor((n_times - 1) / (d - 1))`.

    Returns
    -------
    ndarray, shape (n_times - 1 - (d - 1) * tau, d)
    """
    n_times = x.shape[0]
    tau_max = floor((n_times - 1) / (d - 1))
    if tau > tau_max:
        warn('The given value (%s) for the parameter `tau` exceeds '
             '`tau_max = floor((n_times - 1) / (d - 1))`. Using `tau_max` '
             'instead.' % tau)
        tau = tau_max
    idx = tau * np.arange(d)
    return np.vstack([x[j + idx] for j in range(n_times - 1 - (d - 1) * tau)])


def power_spectrum(sfreq, data, return_db=True):
    """ Utility function to compute the [one sided] Power Spectrum [1, 2].

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    data : ndarray, shape (n_channels, n_times)

    return_db : bool
        If True, the result is returned in dB/Hz.

    Returns
    -------
    ps : ndarray, shape (n_channels, n_times // 2 + 1)

    freqs : ndarray, shape (n_channels, n_times // 2 + 1)
        Array of frequency bins.

    References
    ----------
    .. [1] Heinzel, G. et al. (2002). Spectrum and spectral density estimation
           by the Discrete Fourier transform (DFT), including a comprehensive
           list of window functions and some new at-top windows.

    .. [2] http://fr.mathworks.com/help/signal/ug/power-spectral-density-
           estimates-using-fft.html
    """
    n_times = data.shape[1]
    spect = np.fft.rfft(data, n_times)
    mag = np.abs(spect)
    freqs = np.fft.rfftfreq(n_times, 1. / sfreq)
    ps = np.power(mag, 2) / (n_times ** 2)
    ps *= 2.
    ps[:, 0] /= 2.
    if n_times % 2 == 0:
        ps[:, -1] /= 2.
    if return_db:
        return 10. * np.log10(ps), freqs
    else:
        return ps, freqs
