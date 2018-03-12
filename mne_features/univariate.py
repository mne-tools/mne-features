# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


import numpy as np
from .mock_numba import nb
from math import sqrt
from scipy import stats, signal


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


def _unbiased_autocorr(x):
    """ Unbiased autocorrelation. 
    
    Parameters
    ----------
    x : ndarray, shape (n_times,)
    
    Returns
    -------
    ndarray, shape (2 * n_times + 1,)
    """
    m = x.shape[0] - 1
    lags = np.arange(-m, m + 1)
    s = np.add(m, - np.abs(lags))
    s[np.where(s <= 0)] = 1
    autocorr = signal.fftconvolve(x, x[::-1], mode='full')
    autocorr /= s
    return autocorr


@nb.jit([nb.float64(nb.float64[:], nb.float64[:]),
         nb.float32(nb.float32[:], nb.float32[:])], nopython=True)
def _slope_lstsq(x, y):
    """ Utility function which returns the slope of the linear 
    regression between x and y.  
    
    Parameters
    ----------
    x : ndarray, shape (n_times,)
    
    y : ndarray, shape (n_times,)
    
    Returns
    -------
    float
    """
    n_times = x.shape[0]
    sx2 = 0
    sx = 0
    sy = 0
    sxy = 0
    for j in range(n_times):
        sx2 += x[j] ** 2
        sx += x[j]
        sxy += x[j] * y[j]
        sy += y[j]
    den = n_times * sx2 - (sx ** 2)
    num = n_times * sxy - sx * sy
    return num / den


@nb.jit([nb.float64[:](nb.float64[:]), nb.float32[:](nb.float32[:])],
        nopython=True)
def _accumulate_std(x):
    r = np.zeros((x.shape[0],), dtype=x.dtype)
    for j in range(1, x.shape[0]):
        m = 0
        for k in range(j + 1):
            m += x[k]
        m /= (j + 1)
        s = 0
        for k in range(j + 1):
            s += (x[k] - m) ** 2
        s /= j
        r[j] = sqrt(s)
    return r


def compute_mean(data):
    """ Mean of the data (per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    ndarray, shape (n_channels,)
    """

    ndim = data.ndim
    return np.mean(data, axis=ndim - 1)


def compute_variance(data):
    """ Variance of the data (per channel).

     Parameters
     ----------
     data : shape (n_channels, n_times)

     Returns
     -------
     ndarray, shape (n_channels,)
     """

    ndim = data.ndim
    return np.var(data, axis=ndim - 1, ddof=1)


def compute_ptp(data):
    """ Peak-to-peak (PTP) amplitude of the data (per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    ndarray, shape (n_channels,)
    """
    return np.ptp(data, axis=-1) ** 2


def compute_skewness(data):
    """ Skewness of the data (per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    ndarray, shape (n_channels,)
    """

    ndim = data.ndim
    return stats.skew(data, axis=ndim - 1)


def compute_kurtosis(data):
    """ Kurtosis of the data (per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    ndarray, shape (n_channels,)
    """

    ndim = data.ndim
    return stats.kurtosis(data, axis=ndim - 1, fisher=False)


def compute_hurst_exponent(data):
    """ Hurst exponent [1, 2] of the data (per channel). 
     
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    
    Returns
    -------
    ndarray, shape (n_channels,)
    
    References
    ----------
    .. [1] Devarajan, K. et al. (2014). EEG-Based Epilepsy Detection and 
           Prediction. International Journal of Engineering and Technology, 
           6(3), 212.
    
    .. [2] https://en.wikipedia.org/wiki/Hurst_exponent
    """
    n_channels = data.shape[0]
    hurst_exponent = np.empty((n_channels,))
    for j in range(n_channels):
        m = np.mean(data[j, :])
        y = data[j, :] - m
        z = np.cumsum(y)
        r = (np.maximum.accumulate(z) - np.minimum.accumulate(z))[1:]
        s = _accumulate_std(data[j, :])[1:]
        s[np.where(s == 0)] = 1e-12  # avoid dividing by 0
        y_reg = np.log(r / s)
        x_reg = np.log(np.arange(1, y_reg.shape[0] + 1))
        hurst_exponent[j] = _slope_lstsq(x_reg, y_reg)
    return hurst_exponent.ravel()


def compute_decorrelation_time(sfreq, data):
    """ Decorrelation time (per channel) [1].
     
    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.
        
    data : ndarray, shape (n_channels, n_times)
    
    Returns
    -------
    ndarray, shape (n_channels,)
    
    References
    ----------
    .. [1] Teixeira, C. A. et al. (2011). EPILAB: A software package for 
           studies on the prediction of epileptic seizures. Journal of 
           Neuroscience Methods, 200(2), 257-271.
    """
    n_channels, n_times = data.shape
    decorrelation_times = np.empty((n_channels,))
    for j in range(n_channels):
        ac_channel = _unbiased_autocorr(data[j, :])
        zero_cross = ac_channel[(n_times - 1):] <= 0
        if np.any(zero_cross):
            decorr_time = np.argmax(zero_cross)
            decorr_time /= sfreq
        else:
            decorr_time = -1
        decorrelation_times[j] = decorr_time
    return decorrelation_times


def compute_power_spectrum(sfreq, data, return_db=True):
    """ (One-sided) Power Spectrum of the data. [1, 2] 
    
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
    ps = np.power(mag, 2) / (sfreq * n_times)
    ps[1:] *= 2
    if return_db:
        return 10. * np.log10(ps), freqs
    else:
        return ps, freqs


def compute_spect_hjorth_mobility(sfreq, data, normalize=False):
    """ Hjorth mobility (per channel) [1].
    
    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.
        
    data : ndarray, shape (n_channels, n_times)
    
    normalize : bool (default: False)
        Normalize the result by the total power (see [2]).
    
    Returns
    -------
    ndarray, shape (n_channels,)
    
    References
    ----------
    .. [1] Mormann, F. et al. (2006). Seizure prediction: the long and winding 
           road. Brain, 130(2), 314-333.
           
    .. [2] Teixeira, C. A. et al. (2011). EPILAB: A software package for 
           studies on the prediction of epileptic seizures. Journal of 
           Neuroscience Methods, 200(2), 257-271.
    """
    ps, freqs = compute_power_spectrum(sfreq, data)
    w_freqs = np.power(freqs, 2)
    mobility = np.sum(np.multiply(ps, w_freqs), axis=-1)
    if normalize:
        mobility = np.divide(mobility, np.sum(ps, axis=-1))
    return mobility


def compute_spect_hjorth_complexity(sfreq, data, normalize=False):
    """ Hjorth complexity (per channel) [1].
    
    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.
        
    data : ndarray, shape (n_channels, n_times)
    
    normalize : bool (default: False)
        Normalize the result by the total power (see [2]).
    
    Returns
    -------
    ndarray, shape (n_channels,)
    
    References
    ----------
    .. [1] Mormann, F. et al. (2006). Seizure prediction: the long and winding 
           road. Brain, 130(2), 314-333.
    
    .. [2] Teixeira, C. A. et al. (2011). EPILAB: A software package for 
           studies on the prediction of epileptic seizures. Journal of 
           Neuroscience Methods, 200(2), 257-271.
    """
    ps, freqs = compute_power_spectrum(sfreq, data)
    w_freqs = np.power(freqs, 4)
    complexity = np.sum(np.multiply(ps, w_freqs), axis=-1)
    if normalize:
        complexity = np.divide(complexity, np.sum(ps, axis=-1))
    return complexity
