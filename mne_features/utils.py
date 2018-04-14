# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


""" Utility functions to be used with either univariate or bivariate feature
functions."""

from math import floor
from warnings import warn

import numpy as np
from mne.filter import filter_data

from .mock_numba import nb


@nb.jit()
def triu_idx(n, include_diag=False):
    """ Utility function to generate an enumeration of the pairs of indices
    (i,j) corresponding to the upper triangular part of a (n, n) array.

    Parameters
    ----------
    n : int

    include_diag : bool (default: False)
        If False, the pairs of indices corresponding to the diagonal are
        removed.

    Returns
    -------
    generator
    """
    pos = -1
    for i in range(n):
        for j in range(i + 1 - int(include_diag), n):
            pos += 1
            yield pos, i, j


def embed(x, d, tau):
    """ Utility function to compute the time-delay embedding of a [univariate
    or multivariate] time series x.

    Parameters
    ----------
    x : ndarray, shape (n_channels, n_times)

    d : int
        Embedding dimension.
        The embedding dimension ``d`` should be greater than 2.

    tau : int
        Delay.
        The delay parameter ``tau`` should be less or equal than
        ``floor((n_times - 1) / (d - 1))``.

    Returns
    -------
    output : ndarray, shape (n_channels, n_times - (d - 1) * tau, d)
    """
    tau_max = floor((x.shape[1] - 1) / (d - 1))
    if tau > tau_max:
        warn('The given value (%s) for the parameter `tau` exceeds '
             '`tau_max = floor((n_times - 1) / (d - 1))`. Using `tau_max` '
             'instead.' % tau)
        _tau = tau_max
    else:
        _tau = int(tau)
    x = x.copy()
    X = np.lib.stride_tricks.as_strided(
        x, (x.shape[0], x.shape[1] - d * _tau + _tau, d),
        (x.strides[-2], x.strides[-1], x.strides[-1] * _tau))
    return X


def power_spectrum(sfreq, data, return_db=False):
    """ Utility function to compute the [one sided] Power Spectrum ([Hein02]_,
    [Math]_).

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    data : ndarray, shape (n_channels, n_times)

    return_db : bool (default: False)
        If True, the result is returned in dB/Hz.

    Returns
    -------
    ps : ndarray, shape (n_channels, n_times // 2 + 1)

    freqs : ndarray, shape (n_channels, n_times // 2 + 1)
        Array of frequency bins.

    References
    ----------
    .. [Hein02] Heinzel, G. et al. (2002). Spectrum and spectral density
                estimation by the Discrete Fourier transform (DFT), including
                a comprehensive list of window functions and some new at-top
                windows.

    .. [Math] http://fr.mathworks.com/help/signal/ug/power-spectral-density-
              estimates-using-fft.html
    """
    n_times = data.shape[1]
    m = np.mean(data, axis=-1)
    _data = data - m[:, None]
    spect = np.fft.rfft(_data, n_times)
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


def filt(sfreq, data, filter_freqs, verbose=False):
    """ Utility function to filter data which acts as a wrapper for
    ``mne.filter.filter_data`` ([Mne]_).

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    data : ndarray, shape (n_channels, n_times)

    filter_freqs : array-like, shape (2,)
        Array of cutoff frequencies. If ``filter_freqs[0]`` is None, a low-pass
        filter is used. If ``filter_freqs[1]`` is None, a high-pass filter is
        used. If both ``filter_freqs[0]``, ``filter_freqs[1]`` are not None and
        ``filter_freqs[0] < filter_freqs[1]``, a band-pass filter is used.
        Eventually, if both ``filter_freqs[0]``, ``filter_freqs[1]`` are not
        None and ``filter_freqs[0] > filter_freqs[1]``, a band-stop filter is
        used.

    verbose : bool (default: False)
        Verbosity parameter. If True, info and warnings related to
        ``mne.filter.filter_data`` are printed.

    Returns
    -------
    output : ndarray, shape (n_channels, n_times)

    References
    ----------
    .. [Mne] https://mne-tools.github.io/stable/
    """
    if filter_freqs[0] is None and filter_freqs[1] is None:
        raise ValueError('The values of `filter_freqs` cannot all be None.')
    else:
        _verbose = 40 * (1 - int(verbose))
        return filter_data(data, sfreq=sfreq, l_freq=filter_freqs[0],
                           h_freq=filter_freqs[1], picks=None,
                           fir_design='firwin', verbose=_verbose)
