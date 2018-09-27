# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

"""Utility functions."""

import sys
from functools import partial
from inspect import getmembers, isfunction, getargs
from math import floor
from warnings import warn

import numpy as np
import pywt
from mne.filter import filter_data
from mne.time_frequency import psd_array_welch, psd_array_multitaper

from .mock_numba import nb


@nb.jit()
def _idxiter(n, triu=True, include_diag=False):
    """Enumeration of the upper-triangular part of a squre matrix.

    Utility function to generate an enumeration (C-order) of the pairs of
    indices (i,j) corresponding to the upper triangular part of a (n, n) array.

    Parameters
    ----------
    n : int

    triu : bool (default: True)
        If True, the returned generator is an enumeration of the upper
        triangular part of a (n, n) array. If False, it is an enumeration of
        all the entries in the array (except for diagonal entries if
        ``include_diag`` is False).

    include_diag : bool (default: False)
        If False, the pairs of indices corresponding to the diagonal are
        removed.

    Returns
    -------
    generator
    """
    pos = -1
    for i in range(n):
        for j in range(i * int(triu), n):
            if not include_diag and i == j:
                continue
            else:
                pos += 1
                yield pos, i, j


def _embed(x, d, tau):
    """Time-delay embedding.

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


def power_spectrum(sfreq, data, fmin=0., fmax=256., psd_method='welch',
                   welch_n_fft=256, welch_n_per_seg=None, welch_n_overlap=0,
                   verbose=False):
    """Power Spectral Density (PSD).

    Utility function to compute the (one-sided) Power Spectral Density which
    acts as a wrapper for :func:`mne.time_frequency.psd_array_welch` (if
    ``method='welch'``) or :func:`mne.time_frequency.psd_array_multitaper`
    (if ``method='multitaper'``). The multitaper method, although more
    computationally intensive than Welch's method or FFT, should be prefered
    for 'short' windows. Welch's method is more suitable for 'long' windows.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    data : ndarray, shape (..., n_times).

    fmin : float (default: 0.)
        Lower bound of the frequency range to consider.

    fmax : float (default: 256.)
        Upper bound of the frequency range to consider.

    psd_method : str (default: 'welch')
        Method used to estimate the PSD from the data. The valid values for
        the parameter ``method`` are: ``'welch'``, ``'fft'`` or
        ``'multitaper'``.

    welch_n_fft : int (default: 256)
        The length of the FFT used. The segments will be zero-padded if
        `welch_n_fft > welch_n_per_seg`. This parameter will be ignored if
        `method = 'fft'` or `method = 'multitaper'`.

    welch_n_per_seg : int or None (default: None)
        Length of each Welch segment (windowed with a Hamming window). If
        None, `welch_n_per_seg` is equal to `welch_n_fft`. This parameter
        will be ignored if `method = 'fft'` or `method = 'multitaper'`.

    welch_n_overlap : int (default: 0)
        The number of points of overlap between segments. Should be
        `<= welch_n_per_seg`. This parameter will be ignored if
        `method = 'fft'` or `method = 'multitaper'`.

    verbose : bool (default: False)
        Verbosity parameter. If True, info and warnings related to
        :func:`mne.time_frequency.psd_array_welch` or
        :func:`mne.time_frequency.psd_array_multitaper` are printed.

    Returns
    -------
    psd : ndarray, shape (..., n_freqs)
        Estimated PSD.

    freqs : ndarray, shape (n_freqs,)
        Array of frequency bins.
    """
    _verbose = 40 * (1 - int(verbose))
    _fmin, _fmax = max(0, fmin), min(fmax, sfreq / 2)
    if psd_method == 'welch':
        _n_fft = min(data.shape[-1], welch_n_fft)
        return psd_array_welch(data, sfreq, fmin=_fmin, fmax=_fmax,
                               n_fft=_n_fft, verbose=_verbose,
                               n_per_seg=welch_n_per_seg,
                               n_overlap=welch_n_overlap)
    elif psd_method == 'multitaper':
        return psd_array_multitaper(data, sfreq, fmin=_fmin, fmax=_fmax,
                                    verbose=_verbose)
    elif psd_method == 'fft':
        n_times = data.shape[-1]
        m = np.mean(data, axis=-1)
        _data = data - m[..., None]
        spect = np.fft.rfft(_data, n_times)
        mag = np.abs(spect)
        freqs = np.fft.rfftfreq(n_times, 1. / sfreq)
        psd = np.power(mag, 2) / (n_times ** 2)
        psd *= 2.
        psd[..., 0] /= 2.
        if n_times % 2 == 0:
            psd[..., -1] /= 2.
        mask = np.logical_and(freqs >= _fmin, freqs <= _fmax)
        return psd[..., mask], freqs[mask]
    else:
        raise ValueError('The given method (%s) is not implemented. Valid '
                         'methods for the computation of the PSD are: '
                         '`welch`, `fft` or `multitaper`.' % str(psd_method))


def _filt(sfreq, data, filter_freqs, verbose=False):
    """Filter data.

    Utility function to filter data which acts as a wrapper for
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


def _get_feature_funcs(sfreq, module_name):
    """Inspection for feature functions.

    Inspects a given module and returns a dictionary of feature
    functions in this module. If the module does not contain any feature
    function, an empty dictionary is returned.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    module_name : str
        Name of the module to inspect.

    Returns
    -------
    feature_funcs : dict
    """
    feature_funcs = dict()
    res = getmembers(sys.modules[module_name], isfunction)
    for name, func in res:
        if name.startswith('compute_'):
            alias = name.split('compute_')[-1]
            if hasattr(func, 'func_code'):
                func_code = func.func_code
            else:
                func_code = func.__code__
            args, _, _ = getargs(func_code)
            if 'sfreq' in args[0]:
                feature_funcs[alias] = partial(func, sfreq)
            else:
                feature_funcs[alias] = func
    return feature_funcs


def _wavelet_coefs(data, wavelet_name='db4'):
    """Compute Discrete Wavelet Transform coefficients.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    wavelet_name : str (default: db4)
         Wavelet name (to be used with ``pywt.Wavelet``). The full list of
         Wavelet names are given by: ``[name for family in pywt.families() for
         name in pywt.wavelist(family)]``.

    Returns
    -------
    coefs : list of ndarray
         Coefficients of a DWT (Discrete Wavelet Transform). ``coefs[0]`` is
         the array of approximation coefficient and ``coefs[1:]`` is the list
         of detail coefficients.
    """
    wavelet = pywt.Wavelet(wavelet_name)
    levdec = min(pywt.dwt_max_level(data.shape[-1], wavelet.dec_len), 6)
    coefs = pywt.wavedec(data, wavelet=wavelet, level=levdec)
    return coefs
