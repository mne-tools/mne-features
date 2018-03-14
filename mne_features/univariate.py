# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


from functools import partial
from math import sqrt, log, floor

import numpy as np
from scipy import stats, signal

from .mock_numba import nb


def get_univariate_funcs(sfreq, freq_bands):
    """ Returns a dictionary of univariate feature functions. For each feature
    function, the corresponding key in the dictionary is an alias for the
    function.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    freq_bands : ndarray, shape (n_freqs,)
        Array defining the frequency bands. The j-th frequency band is defined
        as: [freq_bands[j], freq_bands[j + 1]] (0 <= j <= n_freqs - 1).

    Returns
    -------
    univariate_funcs : dict
    """
    univariate_funcs = dict()
    univariate_funcs['mean'] = compute_mean
    univariate_funcs['variance'] = compute_variance
    univariate_funcs['std'] = compute_std
    univariate_funcs['ptp_amplitude'] = compute_ptp
    univariate_funcs['skewness'] = compute_skewness
    univariate_funcs['kurtosis'] = compute_kurtosis
    univariate_funcs['hurst_exp'] = compute_hurst_exponent
    univariate_funcs['decorr_time'] = partial(compute_decorr_time, sfreq)
    univariate_funcs['hjorth_mobility_spect'] = partial(
        compute_spect_hjorth_mobility, sfreq)
    univariate_funcs['hjorth_complexity_spect'] = partial(
        compute_spect_hjorth_complexity, sfreq)
    univariate_funcs['app_entropy'] = compute_app_entropy
    univariate_funcs['samp_entropy'] = compute_samp_entropy
    univariate_funcs['hjorth_mobility'] = compute_hjorth_mobility
    univariate_funcs['hjorth_complexity'] = compute_hjorth_complexity
    univariate_funcs['higuchi_fd'] = compute_higuchi_fd
    univariate_funcs['katz_fd'] = compute_katz_fd
    univariate_funcs['pow_freq_bands'] = partial(
        compute_power_spectrum_freq_bands, sfreq, freq_bands)
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
    return np.mean(data, axis=-1)


def compute_variance(data):
    """ Variance of the data (per channel).

     Parameters
     ----------
     data : shape (n_channels, n_times)

     Returns
     -------
     ndarray, shape (n_channels,)
     """
    return np.var(data, axis=-1, ddof=1)


def compute_std(data):
    """ Standard deviation of the data.

    Parameters
    ----------
    data : shape (n_channels, n_times)

    Returns
    -------
    ndarray, shape (n_channels)
    """
    return np.std(data, axis=-1, ddof=1)


def compute_ptp(data):
    """ Peak-to-peak (PTP) amplitude of the data (per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    ndarray, shape (n_channels,)
    """
    return np.ptp(data, axis=-1)


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


@nb.jit([nb.float64[:](nb.float64[:, :]), nb.float32[:](nb.float32[:, :])],
        nopython=True)
def compute_app_entropy(data):
    """ Approximate Entropy (AppEn, per channel) [1].

    Parameters
    ----------
    data : shape (n_channels, n_times)

    Returns
    -------
    ndarray, shape (n_channels)

    References
    ----------
    .. [1] Teixeira, C. A. et al. (2011). EPILAB: A software package for
           studies on the prediction of epileptic seizures. Journal of
           Neuroscience Methods, 200(2), 257-271.
    """
    n_channels, n_times = data.shape
    appen = np.empty((n_channels,), dtype=data.dtype)
    for t in range(n_channels):
        s = 0
        for j in range(n_times):
            s += data[t, j] ** 2
        s /= (n_times - 1)
        rs = sqrt(s)
        r = 0.25 * rs
        p = 0.
        a = 0.
        b = 0.
        for i in range(n_times - 2):
            for j in range((i + 1), (n_times - 3)):
                d1 = abs(data[t, i] - data[t, j])
                d2 = abs(data[t, i + 1] - data[t, j + 1])
                d3 = abs(data[t, i + 2] - data[t, j + 2])
                if d1 >= d2:
                    da = d1
                else:
                    da = d2
                if da < r:
                    a += 1
                    if d3 < r:
                        b += 1
            if (a > 0) and (b > 0):
                pi = float(b) / float(a)
                p += log(pi)
        appen[t] = (-2.0) * p * (1.0 / (n_times - 2))
    return appen


@nb.jit([nb.float64[:](nb.float64[:, :]), nb.float32[:](nb.float32[:, :])],
        nopython=True)
def compute_samp_entropy(data):
    """ Sample Entropy (SampEn, per channel) [1].

    Parameters
    ----------
    data : shape (n_channels, n_times)

    Returns
    -------
    ndarray, shape (n_channels)

    References
    ----------
    .. [1] Teixeira, C. A. et al. (2011). EPILAB: A software package for
           studies on the prediction of epileptic seizures. Journal of
           Neuroscience Methods, 200(2), 257-271.
    """
    n_channels, n_times = data.shape
    sampen = np.empty((n_channels,), dtype=data.dtype)
    for t in range(n_channels):
        m = 0
        s = 0
        for j in range(n_times):
            m += data[t, j]
            s += data[t, j] ** 2
        m /= n_times
        s /= n_times
        s = sqrt(s)
        x_new = np.zeros(n_times)
        for j in range(n_times):
            x_new[j] = (data[t, j] - m) / s
        mm = 3
        r = 0.2
        lastrun = np.zeros((n_times,))
        run = np.zeros((n_times,))
        a = np.zeros((mm,))
        b = np.zeros((mm,))
        for i in range(n_times - 1):
            nj = n_times - i - 1
            y1 = x_new[i]
            for jj in range(nj):
                j = jj + i + 1
                if abs(x_new[j] - y1) < r:
                    run[jj] = lastrun[jj] + 1
                    m1 = int(min((mm, run[jj])))
                    for k in range(m1):
                        a[k] += 1
                        if j < (n_times - 1):
                            b[k] += 1
                else:
                    run[jj] = 0
            for jj in range(nj):
                lastrun[jj] = run[jj]
        sampen[t] = -log(a[-1] / b[mm - 2])
    return sampen


def compute_decorr_time(sfreq, data):
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
    ps = np.power(mag, 2) / (sfreq * n_times)
    ps[1:] *= 2
    if return_db:
        return 10. * np.log10(ps), freqs
    else:
        return ps, freqs


def compute_power_spectrum_freq_bands(sfreq, freq_bands, data, db=True):
    """ Power Spectrum (computed by frequency bands) [1].

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    freq_bands : ndarray, shape (n_freqs,)
        Array defining the frequency bands. The j-th frequency band is defined
        as: [freq_bands[j], freq_bands[j + 1]] (0 <= j <= n_freqs - 1).

    data : ndarray, shape (n_channels, n_times)

    db : bool (default: True)
        If True, the power spectrum returned by the function
        `compute_power_spectrum` is returned in dB/Hz.

    Returns
    -------
    ndarray, shape (n_channels * (n_freqs - 1),)

    References
    ----------
    .. [1] Teixeira, C. A. et al. (2011). EPILAB: A software package for
           studies on the prediction of epileptic seizures. Journal of
           Neuroscience Methods, 200(2), 257-271.
    """
    n_channels = data.shape[0]
    n_freqs = freq_bands.shape[0]
    ps, freqs = power_spectrum(sfreq, data, return_db=db)
    idx_freq_bands = np.digitize(freqs, freq_bands)
    pow_freq_bands = np.empty((n_channels, n_freqs - 1))
    for j in range(1, n_freqs):
        ps_band = ps[:, idx_freq_bands == j]
        pow_freq_bands[:, j - 1] = np.mean(ps_band, axis=-1)
    return pow_freq_bands.ravel()


def compute_spect_hjorth_mobility(sfreq, data, normalize=False):
    """ Hjorth mobility (computed from the power spectrum, per channel) [1].

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
    ps, freqs = power_spectrum(sfreq, data)
    w_freqs = np.power(freqs, 2)
    mobility = np.sum(np.multiply(ps, w_freqs), axis=-1)
    if normalize:
        mobility = np.divide(mobility, np.sum(ps, axis=-1))
    return mobility


def compute_spect_hjorth_complexity(sfreq, data, normalize=False):
    """ Hjorth complexity (computed from the power spectrum, per channel) [1].

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
    ps, freqs = power_spectrum(sfreq, data)
    w_freqs = np.power(freqs, 4)
    complexity = np.sum(np.multiply(ps, w_freqs), axis=-1)
    if normalize:
        complexity = np.divide(complexity, np.sum(ps, axis=-1))
    return complexity


def compute_hjorth_mobility(data):
    """ Hjorth mobility (computed in the time domain, per channel) [1].

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    ndarray, shape (n_channels,)

    References
    ----------
    .. [1] Paivinen, N. et al. (2005). Epileptic seizure detection: A nonlinear
           viewpoint. Computer methods and programs in biomedicine, 79(2),
           151-159.
    """
    x = np.insert(data, 0, 0, axis=-1)
    dx = np.diff(x, axis=-1)
    sx = np.std(x, ddof=1, axis=-1)
    sdx = np.std(dx, ddof=1, axis=-1)
    mobility = np.divide(sdx, sx)
    return mobility


def compute_hjorth_complexity(data):
    """ Hjorth complexity (computed in the time domain, per channel) [1].

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    ndarray, shape (n_channels,)

    References
    ----------
    .. [1] Paivinen, N. et al. (2005). Epileptic seizure detection: A nonlinear
           viewpoint. Computer methods and programs in biomedicine, 79(2),
           151-159.
    """
    x = np.insert(data, 0, 0, axis=-1)
    dx = np.diff(x, axis=-1)
    m_dx = compute_hjorth_mobility(dx)
    m_x = compute_hjorth_mobility(data)
    complexity = np.divide(m_dx, m_x)
    return complexity


@nb.jit([nb.float64[:](nb.float64[:, :], nb.optional(nb.int64)),
         nb.float32[:](nb.float32[:, :], nb.optional(nb.int32))])
def compute_higuchi_fd(data, kmax=10):
    """ Higuchi Fractal Dimension (per channel) [1, 2].

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    kmax : int (default: 10)
        Maximum delay/offset (in number of samples).

    Returns
    -------
    ndarray, shape (n_channels,)

    References
    ----------
    .. [1] Esteller, R. et al. (2001). A comparison of waveform fractal
           dimension algorithms. IEEE Transactions on Circuits and Systems I:
           Fundamental Theory and Applications, 48(2), 177-183.

    .. [2] Paivinen, N. et al. (2005). Epileptic seizure detection: A nonlinear
           viewpoint. Computer methods and programs in biomedicine, 79(2),
           151-159.
    """
    n_channels, n_times = data.shape
    higuchi = np.empty((n_channels,), dtype=data.dtype)
    for s in range(n_channels):
        lk = np.empty((kmax,))
        x_reg = np.empty((kmax,))
        y_reg = np.empty((kmax,))
        for k in range(1, kmax + 1):
            lm = np.empty((k,))
            for m in range(k):
                ll = 0
                n_max = floor((n_times - m - 1) / k)
                for j in range(1, n_max):
                    ll += abs(data[s, m + j * k] - data[s, m + (j - 1) * k])
                ll /= k
                ll *= (n_times - 1) / (k * n_max)
                lm[m] = ll
            # Mean of lm
            m_lm = 0
            for m in range(k):
                m_lm += lm[m]
            m_lm /= k
            lk[k - 1] = m_lm
            x_reg[k - 1] = log(1. / k)
            y_reg[k - 1] = log(m_lm)
        higuchi[s] = _slope_lstsq(x_reg, y_reg)
    return higuchi


def compute_katz_fd(data):
    """ Katz Fractal Dimension (per channel) [1].

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    ndarray, shape (n_channels,)

    References
    ----------
    .. [1] Esteller, R. et al. (2001). A comparison of waveform fractal
           dimension algorithms. IEEE Transactions on Circuits and Systems I:
           Fundamental Theory and Applications, 48(2), 177-183.
    """
    dists = np.abs(np.diff(data, axis=-1))
    ll = np.sum(dists, axis=-1)
    a = np.mean(dists, axis=-1)
    ln = np.log10(np.divide(ll, a))
    aux_d = data - data[:, 0, None]
    d = np.max(np.abs(aux_d[:, 1:]), axis=-1)
    katz = np.divide(ln, np.add(ln, np.log10(np.divide(d, ll))))
    return katz
