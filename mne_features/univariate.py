# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

"""Univariate feature functions."""

from math import sqrt, log, floor, gamma

import numpy as np
from scipy import stats
from scipy.ndimage import convolve1d
from sklearn.neighbors import KDTree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, explained_variance_score


from .mock_numba import nb
from .utils import (power_spectrum, _embed, _filt, _get_feature_funcs,
                    _get_feature_func_names, _wavelet_coefs, _idxiter,
                    _psd_params_checker)


def get_univariate_funcs(sfreq):
    """Mapping between aliases and univariate feature functions.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    Returns
    -------
    univariate_funcs : dict
    """
    return _get_feature_funcs(sfreq, __name__)


def get_univariate_func_names():
    """List of names of univariate feature functions.

    Returns
    -------
    univariate_func_names : list
    """
    return _get_feature_func_names(__name__)


def _unbiased_autocorr(x, lags=50):
    """Unbiased autocorrelation function.

    The autocorrelation function is computed using the FFT of the signal.

    Parameters
    ----------
    x : ndarray, shape (n_times,)

    lags : int (default: 50)
        Number of lags for the autocorrelation function.

    Returns
    -------
    ndarray, shape (n_lags,)
    """
    n_times = x.shape[0]
    xm = x - np.mean(x)
    dnorm = np.r_[np.arange(1, n_times + 1), np.arange(n_times - 1, 0, -1)]
    fft = np.fft.fft(xm, n=n_times)
    acf = np.fft.ifft(fft * np.conjugate(fft))[:n_times]
    acf /= dnorm[n_times - 1:]
    acf = acf.real
    return acf[:(lags + 1)] / acf[0]


@nb.jit([nb.float64(nb.float64[:], nb.float64[:]),
         nb.float32(nb.float32[:], nb.float32[:])], nopython=True)
def _slope_lstsq(x, y):
    """Slope of a 1D least-squares regression.

    Utility function which returns the slope of the linear regression
    between x and y.

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


@nb.jit([nb.float64[:](nb.float64[:]), nb.float32[:](nb.float32[:])],
        nopython=True)
def _accumulate_max(x):
    r = np.zeros((x.shape[0],), dtype=x.dtype)
    for j in range(x.shape[0]):
        m = -np.inf
        for k in range(j + 1):
            if x[k] >= m:
                m = x[k]
        r[j] = m
    return r


@nb.jit([nb.float64[:](nb.float64[:]), nb.float32[:](nb.float32[:])],
        nopython=True)
def _accumulate_min(x):
    r = np.zeros((x.shape[0],), dtype=x.dtype)
    for j in range(x.shape[0]):
        m = np.inf
        for k in range(j + 1):
            if x[k] <= m:
                m = x[k]
        r[j] = m
    return r


def compute_mean(data):
    """Mean of the data (per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **mean**
    """
    return np.mean(data, axis=-1)


def compute_variance(data):
    """Variance of the data (per channel).

    Parameters
    ----------
    data : shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **variance**
    """
    return np.var(data, axis=-1, ddof=1)


def compute_std(data):
    """Standard deviation of the data.

    Parameters
    ----------
    data : shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels)

    Notes
    -----
    Alias of the feature function: **std**
    """
    return np.std(data, axis=-1, ddof=1)


def compute_ptp_amp(data):
    """Peak-to-peak (PTP) amplitude of the data (per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **ptp_amp**
    """
    return np.ptp(data, axis=-1)


def compute_skewness(data):
    """Skewness of the data (per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **skewness**
    """
    ndim = data.ndim
    return stats.skew(data, axis=ndim - 1)


def compute_kurtosis(data):
    """Kurtosis of the data (per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **kurtosis**
    """
    ndim = data.ndim
    return stats.kurtosis(data, axis=ndim - 1, fisher=False)


@nb.jit([nb.float64[:, :](nb.float64[:, :]),
         nb.float32[:, :](nb.float32[:, :])], nopython=True)
def _hurst_exp_compute_rs(x):
    """Utility function for :func:`compute_hurst_exp`.

    Parameters
    ----------
    x : ndarray, shape (n_seqs, n_times)

    Returns
    -------
    output : ndarray, shape (n_seqs, n_times - 1)
    """
    n_seqs, n_times = x.shape
    rs = np.zeros((n_seqs, n_times - 1), dtype=x.dtype)
    for j in range(n_seqs):
        m = 0
        for k in range(n_times):
            m += x[j, k]
        m /= n_times
        y = np.empty((n_times,))
        for k in range(n_times):
            y[k] = x[j, k] - m
        z = np.empty((n_times,))
        z[0] = y[0]
        for k in range(1, n_times):
            z[k] = z[k - 1] + y[k]
        r = _accumulate_max(z) - _accumulate_min(z)
        s = _accumulate_std(x[j, :])
        for k in range(1, n_times):
            if s[k] == 0:
                rs[j, k - 1] = np.nan
            else:
                rs[j, k - 1] = r[k] / s[k]
    return rs


def _hurst_exp_helper(x, n_splits=20):
    """Helper function for :func:`compute_hurst_exp`.

    Compute the Hurst exponent from a univariate time series. The Hurst
    exponent is defined as the slope of the least-squares regression line
    going through a cloud of `n_splits` points. Each point is obtained by
    considering sub-series of `x` of `n_splits` different lenghts.

    Parameters
    ----------
    x : ndarray, shape (n_times,)

    Returns
    -------
    output : ndarray, shape (n_splits,)
    """
    n_times = x.shape[0]
    _splits = np.floor(np.logspace(start=4, stop=np.log2(n_times / 2),
                                   num=n_splits, base=2.))
    splits = np.unique(_splits).astype(int)
    reg = np.zeros((splits.size,))
    for j, n in enumerate(splits):
        a = x.copy()
        d = int(floor(n_times / n))
        a = np.lib.stride_tricks.as_strided(a, shape=(d, n),
                                            strides=(n * a.strides[-1],
                                                     a.strides[-1]))
        _rs = _hurst_exp_compute_rs(a)
        _rs = _rs[~np.isnan(_rs)]
        reg[j] = np.log(np.mean(_rs))
        s = sum([sqrt((n - i) / i) for i in range(1, n)]) * ((n - 0.5) / n)
        if n <= 340:
            corr = (gamma((n - 1) / 2.) / (sqrt(np.pi) * gamma(n / 2.))) * s
        else:
            corr = ((n - 0.5) / n) * (1. / sqrt(np.pi * n / 2.)) * s
        reg[j] -= (np.log(corr) - np.log(n) / 2)
    return _slope_lstsq(np.log(splits), reg)


def compute_hurst_exp(data):
    """Hurst exponent of the data (per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **hurst_exp**. See [1]_ and [2]_.

    References
    ----------
    .. [1] Rasheed, B. Q. K. et al. (2004). Hurst exponent and financial
           market predictability. In IASTED conference on Financial
           Engineering and Applications (FEA 2004) (pp. 203-209).

    .. [2] Devarajan, K. et al. (2014). EEG-Based Epilepsy Detection and
           Prediction. International Journal of Engineering and Technology,
           6(3), 212.
    """
    n_channels, n_times = data.shape
    hurst = np.empty((n_channels,))
    for j in range(n_channels):
        hurst[j] = _hurst_exp_helper(data[j, :])
    return hurst


def _app_samp_entropy_helper(data, emb, metric='chebyshev',
                             approximate=True):
    """Utility function for `compute_app_entropy`` and `compute_samp_entropy`.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    emb : int (default: 2)
        Embedding dimension.

    metric : str (default: chebyshev)
        Name of the metric function used with KDTree. The list of available
        metric functions is given by: ``KDTree.valid_metrics``.

    approximate : bool (default: True)
        If True, the returned values will be used to compute the
        Approximate Entropy (AppEn). Otherwise, the values are used to compute
        the Sample Entropy (SampEn).

    Returns
    -------
    output : ndarray, shape (n_channels, 2)
    """
    _all_metrics = KDTree.valid_metrics
    if metric not in _all_metrics:
        raise ValueError('The given metric (%s) is not valid. The valid '
                         'metric names are: %s' % (metric, _all_metrics))
    n_channels, n_times = data.shape
    phi = np.empty((n_channels, 2))
    for j in range(n_channels):
        r = 0.2 * np.std(data[j, :], axis=-1, ddof=1)
        # compute phi(emb, r)
        _emb_data1 = _embed(data[j, None], emb, 1)[0, :, :]
        if approximate:
            emb_data1 = _emb_data1
        else:
            emb_data1 = _emb_data1[:-1, :]
        count1 = KDTree(emb_data1, metric=metric).query_radius(
            emb_data1, r, count_only=True).astype(np.float64)
        # compute phi(emb + 1, r)
        emb_data2 = _embed(data[j, None], emb + 1, 1)[0, :, :]
        count2 = KDTree(emb_data2, metric=metric).query_radius(
            emb_data2, r, count_only=True).astype(np.float64)
        if approximate:
            phi[j, 0] = np.mean(np.log(count1 / emb_data1.shape[0]))
            phi[j, 1] = np.mean(np.log(count2 / emb_data2.shape[0]))
        else:
            phi[j, 0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
            phi[j, 1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
    return phi


def compute_app_entropy(data, emb=2, metric='chebyshev'):
    """Approximate Entropy (AppEn, per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    emb : int (default: 2)
        Embedding dimension.

    metric : str (default: chebyshev)
        Name of the metric function used with
        :class:`~sklearn.neighbors.KDTree`. The list of available
        metric functions is given by: ``KDTree.valid_metrics``.

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **app_entropy**. See [1]_.

    References
    ----------
    .. [1] Richman, J. S. et al. (2000). Physiological time-series analysis
           using approximate entropy and sample entropy. American Journal of
           Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
    """
    phi = _app_samp_entropy_helper(data, emb=emb, metric=metric,
                                   approximate=True)
    return np.subtract(phi[:, 0], phi[:, 1])


def compute_samp_entropy(data, emb=2, metric='chebyshev'):
    """Sample Entropy (SampEn, per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    emb : int (default: 2)
        Embedding dimension.

    metric : str (default: chebyshev)
        Name of the metric function used with KDTree. The list of available
        metric functions is given by: `KDTree.valid_metrics`.

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **samp_entropy**. See [1]_.

    References
    ----------
    .. [1] Richman, J. S. et al. (2000). Physiological time-series analysis
           using approximate entropy and sample entropy. American Journal of
           Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
    """
    phi = _app_samp_entropy_helper(data, emb=emb, metric=metric,
                                   approximate=False)
    if np.allclose(phi[:, 0], 0) or np.allclose(phi[:, 1], 0):
        raise ValueError('Sample Entropy is not defined.')
    else:
        return -np.log(np.divide(phi[:, 1], phi[:, 0]))


def compute_decorr_time(sfreq, data):
    """Decorrelation time (per channel).

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **decorr_time**. See [1]_.

    References
    ----------
    .. [1] Teixeira, C. A. et al. (2011). EPILAB: A software package for
           studies on the prediction of epileptic seizures. Journal of
           Neuroscience Methods, 200(2), 257-271.
    """
    n_channels = data.shape[0]
    decorrelation_times = np.empty((n_channels,))
    for j in range(n_channels):
        _acf = _unbiased_autocorr(data[j, :])
        zc = np.diff(np.sign(_acf)) != 0
        if np.any(zc):
            decorr_time = np.argmax(zc) + 1
            decorr_time /= sfreq
        else:
            decorr_time = -1
        decorrelation_times[j] = decorr_time
    return decorrelation_times


def _freq_bands_helper(sfreq, freq_bands):
    """Utility function to define frequency bands.

    This utility function is to be used with :func:`compute_pow_freq_bands` and
    :func:`compute_energy_freq_bands`. It essentially checks if the given
    parameter ``freq_bands`` is valid and raises an error if not.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    freq_bands : ndarray, shape (n_freq_bands + 1,) or (n_freq_bands, 2)
        Array defining frequency bands.

    Returns
    -------
    valid_freq_bands : ndarray, shape (n_freq_bands, 2)
    """
    if not np.logical_and(freq_bands >= 0, freq_bands <= sfreq / 2).all():
        raise ValueError('The entries of the given `freq_bands` parameter '
                         '(%s) must be positive and less than the Nyquist '
                         'frequency.' % str(freq_bands))
    else:
        if freq_bands.ndim == 1:
            n_freq_bands = freq_bands.shape[0] - 1
            valid_freq_bands = np.empty((n_freq_bands, 2))
            for j in range(n_freq_bands):
                valid_freq_bands[j, :] = freq_bands[j:j + 2]
        elif freq_bands.ndim == 2 and freq_bands.shape[-1] == 2:
            valid_freq_bands = freq_bands
        else:
            raise ValueError('The given value (%s) for the `freq_bands` '
                             'parameter is not valid. Only 1D or 2D arrays '
                             'with shape (n_freq_bands, 2) are accepted.'
                             % str(freq_bands))
        return valid_freq_bands


def compute_pow_freq_bands(sfreq, data, freq_bands=np.array([0.5, 4., 8., 13.,
                                                             30., 100.]),
                           normalize=True, ratios=None, ratios_triu=False,
                           psd_method='welch', log=False, psd_params=None):
    """Power Spectrum (computed by frequency bands).

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    data : ndarray, shape (n_channels, n_times)

    freq_bands : ndarray or dict (default: np.array([.5, 4, 8, 13, 30, 100]))
        The parameter ``freq_bands`` should be either a ndarray with shape
        ``(n_freq_bands + 1,)`` or ``(n_freq_bands, 2)`` or a dict. If ndarray
        with shape ``(n_freq_bands + 1,)``, the entries define **contiguous**
        frequency bands as follows: the i-th frequency band is defined as:
        [freq_bands[i], freq_bands[i + 1]] (0 <= i <= n_freq_bands - 1). If
        ndarray with shape ``(n_freq_bands, 2)``, the rows of ``freq_bands``
        define **non-contiguous** frequency bands. If dict, the keys should be
        strings (names of the frequency bands) and the values, the
        corresponding bands (as ndarray with shape (2,) or list of length 2).
        When ``freq_bands`` is of type dict, the keys are used to generate the
        feature names (only used when features are extracted with
        ``return_as_df=True``). The values of ``freq_bands`` should be between
        0 and sfreq / 2 (the Nyquist frequency) as the function uses the
        one-sided PSD.

    normalize : bool (default: True)
        If True, the average power in each frequency band is normalized by
        the total power.

    ratios : str or None (default: None)
        If not None, the possible values for the parameter ``ratios`` are:
        ``all`` or ``only``. If ``all``, the function will return the power
        (computed in the given frequency bands) as well as the ratios between
        power in different frequency bands. Depending on the value of
        ``ratios_triu``, either all possible pairs of distinct frequency bands
        are considered, or only the upper triangle of the matrix of pairs.
        If ``only``, the function returns only the ratios of power in bands. If
        None, no ratio is computed.

    ratios_triu : bool (default: False)
        If False, include all possible pairs when computing ratios
        (e.g., alpha/beta and beta/alpha) yielding
        n_freq_bands * (n_freq_bands - 1) ratios.
        If True, only include the (upper) triangle of the square matrix,
        yielding n_freq_bands * (n_freq_bans - 1) / 2 ratios. Ignored if
        ``ratios`` is None.

    psd_method : str (default: 'welch')
        Method used for the estimation of the Power Spectral Density (PSD).
        Valid methods are: ``'welch'``, ``'multitaper'`` or ``'fft'``.

    log : bool (default: False)
        If True, the average power in each frequency band is transformed with a
        base-10 logaritm. Ratios (computed if ``ratios`` is specified) will be
        computed as log-ratios, i.e., the ratio of bands A and B will be
        `log(pow_band_A / pow_band_B) = log_pow_band_A - log_pow_band_B`.

    psd_params : dict or None (default: None)
        If not None, dict with optional parameters (`welch_n_fft`,
        `welch_n_per_seg`, `welch_n_overlap`) to be passed to
        :func:`mne_features.utils.power_spectrum`. If None, default parameters
        are used (see doc for :func:`mne_features.utils.power_spectrum`).

    Returns
    -------
    output : ndarray, shape (n_channels * (n_freqs - 1),)

    Notes
    -----
    Alias of the feature function: **pow_freq_bands**. See [1]_.

    References
    ----------
    .. [1] Teixeira, C. A. et al. (2011). EPILAB: A software package for
           studies on the prediction of epileptic seizures. Journal of
           Neuroscience Methods, 200(2), 257-271.
    """
    n_channels = data.shape[0]
    if isinstance(freq_bands, dict):
        _freq_bands = np.asarray([freq_bands[n] for n in freq_bands])
    else:
        _freq_bands = np.asarray(freq_bands)
    fb = _freq_bands_helper(sfreq, _freq_bands)
    n_freq_bands = fb.shape[0]
    _psd_params = _psd_params_checker(psd_params)
    psd, freqs = power_spectrum(sfreq, data, psd_method=psd_method,
                                **_psd_params)
    pow_freq_bands = np.empty((n_channels, n_freq_bands))
    for j in range(n_freq_bands):
        mask = np.logical_and(freqs >= fb[j, 0], freqs <= fb[j, 1])
        psd_band = psd[:, mask]
        pow_freq_bands[:, j] = np.sum(psd_band, axis=-1)

    if log:
        pow_freq_bands = np.log10(pow_freq_bands)

    if normalize:
        pow_freq_bands = np.divide(pow_freq_bands,
                                   np.sum(psd, axis=-1)[:, None])
    if ratios is None:
        return pow_freq_bands.ravel()
    elif ratios not in ['all', 'only']:
        raise ValueError('The given value (%s) for the parameter `ratios` '
                         'is not valid. Valid values are: `all` or `only`.'
                         % str(ratios))
    else:
        n_ratios = n_freq_bands * (n_freq_bands - 1) // (1 + int(ratios_triu))
        band_ratios = np.empty((n_channels, n_ratios))
        for pos, i, j in _idxiter(n_freq_bands, triu=ratios_triu):
            if log:
                band_ratios[:, pos] = \
                    pow_freq_bands[:, i] - pow_freq_bands[:, j]
            else:
                band_ratios[:, pos] = \
                    pow_freq_bands[:, i] / pow_freq_bands[:, j]
        if ratios == 'all':
            return np.r_[pow_freq_bands.ravel(), band_ratios.ravel()]
        else:
            return band_ratios.ravel()


def _compute_pow_freq_bands_feat_names(data, freq_bands, normalize, ratios,
                                       ratios_triu, psd_method, log,
                                       psd_params):
    """Utility function to create feature names compatible with the output
    of :func:`compute_pow_freq_bands`."""
    n_channels = data.shape[0]
    if isinstance(freq_bands, dict):
        n_freq_bands = len(freq_bands)
        _band_names = [str(n) for n in freq_bands]
    else:
        freq_bands = np.array(freq_bands)  # Cast in case it's a list of lists
        n_freq_bands = (freq_bands.shape[0] - 1 if freq_bands.ndim == 1
                        else freq_bands.shape[0])
        _band_names = ['band' + str(j) for j in range(n_freq_bands)]
    ratios_names = ['ch%s_%s/%s' % (ch, _band_names[i], _band_names[j])
                    for ch in range(n_channels) for _, i, j in
                    _idxiter(n_freq_bands, triu=ratios_triu)]
    pow_names = ['ch%s_%s' % (ch, _band_names[i]) for ch in
                 range(n_channels) for i in range(n_freq_bands)]
    if ratios is None:
        return pow_names
    elif ratios == 'only':
        return ratios_names
    else:
        return pow_names + ratios_names


compute_pow_freq_bands.get_feature_names = _compute_pow_freq_bands_feat_names


def compute_hjorth_mobility_spect(sfreq, data, normalize=False,
                                  psd_method='welch', psd_params=None):
    """Hjorth mobility (per channel).

    Hjorth mobility parameter computed from the Power Spectrum of the data.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    data : ndarray, shape (n_channels, n_times)

    normalize : bool (default: False)
        Normalize the result by the total power.

    psd_method : str (default: 'welch')
        Method used for the estimation of the Power Spectral Density (PSD).
        Valid methods are: ``'welch'``, ``'multitaper'`` or ``'fft'``.

    psd_params : dict or None (default: None)
        If not None, dict with optional parameters (`welch_n_fft`,
        `welch_n_per_seg`, `welch_n_overlap`) to be passed to
        :func:`mne_features.utils.power_spectrum`. If None, default parameters
        are used (see doc for :func:`mne_features.utils.power_spectrum`).

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **hjorth_mobility_spect**. See [1]_ and
    [2]_.

    References
    ----------
    .. [1] Mormann, F. et al. (2006). Seizure prediction: the long and
           winding road. Brain, 130(2), 314-333.

    .. [2] Teixeira, C. A. et al. (2011). EPILAB: A software package for
           studies on the prediction of epileptic seizures. Journal of
           Neuroscience Methods, 200(2), 257-271.
    """
    _psd_params = _psd_params_checker(psd_params)
    psd, freqs = power_spectrum(sfreq, data, psd_method=psd_method,
                                **_psd_params)
    w_freqs = np.power(freqs, 2)
    mobility = np.sum(np.multiply(psd, w_freqs), axis=-1)
    if normalize:
        mobility = np.divide(mobility, np.sum(psd, axis=-1))
    return mobility


def compute_hjorth_complexity_spect(sfreq, data, normalize=False,
                                    psd_method='welch', psd_params=None):
    """Hjorth complexity (per channel).

    Hjorth complexity parameter computed from the Power Spectrum of the data.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    data : ndarray, shape (n_channels, n_times)

    normalize : bool (default: False)
        Normalize the result by the total power.

    psd_method : str (default: 'welch')
        Method used for the estimation of the Power Spectral Density (PSD).
        Valid methods are: ``'welch'``, ``'multitaper'`` or ``'fft'``.

    psd_params : dict or None (default: None)
        If not None, dict with optional parameters (`welch_n_fft`,
        `welch_n_per_seg`, `welch_n_overlap`) to be passed to
        :func:`mne_features.utils.power_spectrum`. If None, default parameters
        are used (see doc for :func:`mne_features.utils.power_spectrum`).

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **hjorth_complexity_spect**. See [1]_ and
    [2]_.

    References
    ----------
    .. [1] Mormann, F. et al. (2006). Seizure prediction: the long and
           winding road. Brain, 130(2), 314-333.

    .. [2] Teixeira, C. A. et al. (2011). EPILAB: A software package for
           studies on the prediction of epileptic seizures. Journal of
           Neuroscience Methods, 200(2), 257-271.
    """
    _psd_params = _psd_params_checker(psd_params)
    psd, freqs = power_spectrum(sfreq, data, psd_method=psd_method,
                                **_psd_params)
    w_freqs = np.power(freqs, 4)
    complexity = np.sum(np.multiply(psd, w_freqs), axis=-1)
    if normalize:
        complexity = np.divide(complexity, np.sum(psd, axis=-1))
    return complexity


def compute_hjorth_mobility(data):
    """Hjorth mobility (per channel).

    Hjorth mobility parameter computed in the time domain.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **hjorth_mobility**. See [1]_.

    References
    ----------
    .. [1] Paivinen, N. et al. (2005). Epileptic seizure detection: A
           nonlinear viewpoint. Computer methods and programs in biomedicine,
           79(2), 151-159.
    """
    x = np.insert(data, 0, 0, axis=-1)
    dx = np.diff(x, axis=-1)
    sx = np.std(x, ddof=1, axis=-1)
    sdx = np.std(dx, ddof=1, axis=-1)
    mobility = np.divide(sdx, sx)
    return mobility


def compute_hjorth_complexity(data):
    """Hjorth complexity (per channel).

    Hjorth complexity parameter computed in the time domain.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **hjorth_complexity**. See [1]_.

    References
    ----------
    .. [1] Paivinen, N. et al. (2005). Epileptic seizure detection: A
           nonlinear viewpoint. Computer methods and programs in biomedicine,
           79(2), 151-159.
    """
    x = np.insert(data, 0, 0, axis=-1)
    dx = np.diff(x, axis=-1)
    m_dx = compute_hjorth_mobility(dx)
    m_x = compute_hjorth_mobility(data)
    complexity = np.divide(m_dx, m_x)
    return complexity


@nb.jit([nb.float64[:](nb.float64[:, :], nb.int64),
         nb.float32[:](nb.float32[:, :], nb.int32)], nopython=True)
def _higuchi_fd(data, kmax):
    """Utility function for :func:`compute_higuchi_fd`.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    kmax : int

    Returns
    -------
    output : ndarray, shape (n_channels,)
    """
    n_channels, n_times = data.shape
    higuchi = np.empty((n_channels,), dtype=data.dtype)
    for s in range(n_channels):
        kmax = np.int64(kmax)
        lk = np.empty((kmax,))
        x_reg = np.empty((kmax,))
        y_reg = np.empty((kmax,))
        for k in range(1, kmax + 1):
            lm = np.empty((k,))
            for m in range(k):
                ll = 0
                n_max = floor((n_times - m - 1) / k)
                n_max = int(n_max)
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


def compute_higuchi_fd(data, kmax=10):
    """Higuchi Fractal Dimension (per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    kmax : int (default: 10)
        Maximum delay/offset (in number of samples).

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **higuchi_fd**. See [1]_ and [2]_.

    References
    ----------
    .. [1] Esteller, R. et al. (2001). A comparison of waveform fractal
           dimension algorithms. IEEE Transactions on Circuits and Systems I:
           Fundamental Theory and Applications, 48(2), 177-183.

    .. [2] Paivinen, N. et al. (2005). Epileptic seizure detection: A
           nonlinear viewpoint. Computer methods and programs in biomedicine,
           79(2), 151-159.
    """
    return _higuchi_fd(data, kmax)


def compute_katz_fd(data):
    """Katz Fractal Dimension (per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **katz_fd**. See [1]_.

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


def compute_zero_crossings(data, threshold=np.finfo(np.float64).eps):
    """Number of zero-crossings (per channel).

    The ``threshold`` parameter is used to clip 'small' values to zero.
    Changing its default value is likely to affect the number of
    zero-crossings returned by the function.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    threshold : float (default: np.finfo(np.float64).eps)
        Threshold used to determine when a float should de treated as zero.

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **zero_crossings**
    """
    _data = data.copy()
    # clip 'small' values to 0
    _data[np.abs(_data) < threshold] = 0
    sgn = np.sign(_data)
    # sgn may already contain 0 values (either 'true' zeros or clipped values)
    aux = np.diff((sgn == 0).astype(np.int64), axis=-1)
    count = np.sum(aux == 1, axis=-1) + (_data[:, 0] == 0)
    # zero between two consecutive time points (data[i] * data[i + 1] < 0)
    mask_implicit_zeros = sgn[:, 1:] * sgn[:, :-1] < 0
    count += np.sum(mask_implicit_zeros, axis=-1)
    return count


def compute_line_length(data):
    """Line length (per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **line_length**. See [1]_.

    References
    ----------
    .. [1] Esteller, R. et al. (2001). Line length: an efficient feature for
           seizure onset detection. In Engineering in Medicine and Biology
           Society, 2001. Proceedings of the 23rd Annual International
           Conference of the IEEE (Vol. 2, pp. 1707-1710). IEEE.
    """
    return np.mean(np.abs(np.diff(data, axis=-1)), axis=-1)


def compute_spect_entropy(sfreq, data, psd_method='welch', psd_params=None):
    """Spectral Entropy (per channel).

    Spectral Entropy is defined to be the Shannon Entropy of the Power
    Spectrum of the data.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data

    data : ndarray, shape (n_channels, n_times)

    psd_method : str (default: 'welch')
        Method used for the estimation of the Power Spectral Density (PSD).
        Valid methods are: ``'welch'``, ``'multitaper'`` or ``'fft'``.

    psd_params : dict or None (default: None)
        If not None, dict with optional parameters (`welch_n_fft`,
        `welch_n_per_seg`, `welch_n_overlap`) to be passed to
        :func:`mne_features.utils.power_spectrum`. If None, default parameters
        are used (see doc for :func:`mne_features.utils.power_spectrum`).

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **spect_entropy**. See [1]_.

    References
    ----------
    .. [1] Inouye, T. et al. (1991). Quantification of EEG irregularity by
           use of the entropy of the power spectrum. Electroencephalography
           and clinical neurophysiology, 79(3), 204-210.
    """
    _psd_params = _psd_params_checker(psd_params)
    psd, _ = power_spectrum(sfreq, data, psd_method=psd_method, **_psd_params)
    m = np.sum(psd, axis=-1)
    psd_norm = np.divide(psd[:, 1:], m[:, None])
    return -np.sum(np.multiply(psd_norm, np.log2(psd_norm)), axis=-1)


def compute_svd_entropy(data, tau=2, emb=10):
    """SVD entropy (per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    tau : int (default: 2)
        Delay (number of samples).

    emb : int (default: 10)
        Embedding dimension.

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **svd_entropy**. See [1]_.

    References
    ----------
    .. [1] Roberts, S. J. et al. (1999). Temporal and spatial complexity
           measures for electroencephalogram based brain-computer interfacing.
           Medical & biological engineering & computing, 37(1), 93-98.
    """
    _, sv, _ = np.linalg.svd(_embed(data, d=emb, tau=tau))
    m = np.sum(sv, axis=-1)
    sv_norm = np.divide(sv, m[:, None])
    return -np.sum(np.multiply(sv_norm, np.log2(sv_norm)), axis=-1)


def compute_spect_slope(sfreq, data, fmin=0.1, fmax=50,
                        with_intercept=True, psd_method='welch',
                        psd_params=None):
    """Linear regression of the the log-log frequency-curve (per channel).

    Using a linear regression, the function estimates the slope and the
    intercept (if ``with_intercept`` is True) of the Power Spectral Density
    (PSD) in the log-log scale. In addition to this, the Mean Square Error
    (MSE) and R2 coefficient (goodness-of-fit) are returned. By default, the
    [0.1Hz, 50Hz] frequency range is used for the regression.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    data : ndarray, shape (n_channels, n_times)

    fmin : float (default: 0.1)
        Lower bound of the frequency range considered in the linear regression.

    fmax : float (default: 50)
        Upper bound of the frequency range considered in the linear regression.

    with_intercept : bool (default: True)
        If True, the intercept of the linear regression is included among the
        features returned by the function. If False, only the slope, the MSE
        and the R2 coefficient are returned.

    psd_method : str (default: 'welch')
        Method used for the estimation of the Power Spectral Density (PSD).
        Valid methods are: ``'welch'``, ``'multitaper'`` or ``'fft'``.

    psd_params : dict or None (default: None)
        If not None, dict with optional parameters (`welch_n_fft`,
        `welch_n_per_seg`, `welch_n_overlap`) to be passed to
        :func:`mne_features.utils.power_spectrum`. If None, default parameters
        are used (see doc for :func:`mne_features.utils.power_spectrum`).

    Returns
    -------
    output : ndarray, shape (n_channels * 4,)
        The four characteristics: intercept, slope, MSE, and R2 per channel.

    Notes
    -----
    Alias of the feature function: **spect_slope**. See [1]_
    and [2]_.

    References
    ----------
    .. [1] Demanuelle C. et al. (2007). Distinguishing low frequency
           oscillations within the 1/f spectral behaviour of electromagnetic
           brain signals. Behavioral and Brain Functions (BBF).

    .. [2] Winkler I. et al. (2011). Automatic Classification of Artifactual
           ICA-Components for Artifact Removal in EEG Signals. Behavioral and
           Brain Functions (BBF).
    """
    n_channels = data.shape[0]
    _psd_params = _psd_params_checker(psd_params)
    psd, freqs = power_spectrum(sfreq, data, psd_method=psd_method,
                                **_psd_params)

    # mask limiting to input freq_range
    mask = np.logical_and(freqs >= fmin, freqs <= fmax)

    # freqs and psd selected over input freq_range and expressed in log scale
    freqs, psd = np.log10(freqs[mask]), np.log10(psd[:, mask])

    # linear fit
    lm = LinearRegression()
    fit_info = np.empty((n_channels, 4))
    for idx, power in enumerate(psd):
        lm.fit(freqs.reshape(-1, 1), power)
        fit_info[idx, 0] = lm.intercept_
        fit_info[idx, 1] = lm.coef_
        power_estimate = lm.predict(freqs.reshape(-1, 1))
        fit_info[idx, 2] = mean_squared_error(power, power_estimate)
        fit_info[idx, 3] = explained_variance_score(power, power_estimate)
    if not with_intercept:
        fit_info = fit_info[:, 1:]
    return fit_info.ravel()


def _compute_spect_slope_feat_names(data, **kwargs):
    """Utility function to create feature names compatible with the output of
    :func:`mne_features.univariate.compute_energy_freq_bands`."""
    n_channels = data.shape[0]
    stats = ['intercept', 'slope', 'MSE', 'R2']
    return ['ch%s_%s' % (ch, stat) for ch in range(n_channels)
            for stat in stats]


compute_spect_slope.get_feature_names = _compute_spect_slope_feat_names


def compute_svd_fisher_info(data, tau=2, emb=10):
    """SVD Fisher Information (per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    tau : int (default: 2)
        Delay (number of samples).

    emb : int (default: 10)
        Embedding dimension.

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **svd_fisher_info**. See [1]_.

    References
    ----------
    .. [1] Roberts, S. J. et al. (1999). Temporal and spatial complexity
           measures for electroencephalogram based brain-computer interfacing.
           Medical & biological engineering & computing, 37(1), 93-98.
    """
    _, sv, _ = np.linalg.svd(_embed(data, d=emb, tau=tau))
    m = np.sum(sv, axis=-1)
    sv_norm = np.divide(sv, m[:, None])
    aux = np.divide(np.diff(sv_norm, axis=-1) ** 2, sv_norm[:, :-1])
    return np.sum(aux, axis=-1)


def compute_energy_freq_bands(sfreq, data, freq_bands=np.array([0.5, 4., 8.,
                                                                13., 30.,
                                                                100.]),
                              deriv_filt=True):
    """Band energy (per channel).

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    data : ndarray, shape (n_channels, n_times)

    freq_bands : ndarray or dict (default: np.array([.5, 4, 8, 13, 30, 100]))
        The parameter ``freq_bands`` should be either a ndarray with shape
        ``(n_freq_bands + 1,)`` or ``(n_freq_bands, 2)`` or a dict. If ndarray
        with shape ``(n_freq_bands + 1,)``, the entries define **contiguous**
        frequency bands as follows: the i-th frequency band is defined as:
        [freq_bands[i], freq_bands[i + 1]] (0 <= i <= n_freq_bands - 1). If
        ndarray with shape ``(n_freq_bands, 2)``, the rows of ``freq_bands``
        define **non-contiguous** frequency bands. If dict, the keys should be
        strings (names of the frequency bands) and the values, the
        corresponding bands (as ndarray with shape (2,) or list of length 2).
        When ``freq_bands`` is of type dict, the keys are used to generate the
        feature names (only used when features are extracted with
        ``return_as_df=True``). The values of ``freq_bands`` should be between
        0 and sfreq / 2 (the Nyquist frequency) as the function uses the
        one-sided PSD.

    deriv_filt : bool (default: False)
        If True, a derivative filter is applied to the input data before
        filtering (see Notes).

    Returns
    -------
    output : ndarray, shape (n_channels * (n_freqs - 1),)

    Notes
    -----
    Alias of the feature function: **energy_freq_bands**. See [1]_.

    References
    ----------
    .. [1] Kharbouch, A. et al. (2011). An algorithm for seizure onset
           detection using intracranial EEG. Epilepsy & Behavior, 22, S29-S35.
    """
    n_channels = data.shape[0]
    if isinstance(freq_bands, dict):
        _freq_bands = np.asarray([freq_bands[n] for n in freq_bands])
    else:
        _freq_bands = np.asarray(freq_bands)
    fb = _freq_bands_helper(sfreq, _freq_bands)
    n_freq_bands = fb.shape[0]
    band_energy = np.empty((n_channels, n_freq_bands))
    if deriv_filt:
        _data = convolve1d(data, [1., 0., -1.], axis=-1, mode='nearest')
    else:
        _data = data
    for j in range(n_freq_bands):
        filtered_data = _filt(sfreq, _data, fb[j, :])
        band_energy[:, j] = np.sum(filtered_data ** 2, axis=-1)
    return band_energy.ravel()


def _compute_energy_fb_feat_names(data, freq_bands, deriv_filt):
    """Utility function to create feature names compatible with the output of
    :func:`mne_features.univariate.compute_energy_freq_bands`."""
    n_channels = data.shape[0]
    if isinstance(freq_bands, dict):
        n_freq_bands = len(freq_bands)
        _band_names = [str(n) for n in freq_bands]
    else:
        n_freq_bands = (freq_bands.shape[0] - 1 if freq_bands.ndim == 1
                        else freq_bands.shape[0])
        _band_names = ['band' + str(j) for j in range(n_freq_bands)]
    return ['ch%s_%s' % (ch, _band_names[i]) for ch in range(n_channels)
            for i in range(n_freq_bands)]


compute_energy_freq_bands.get_feature_names = _compute_energy_fb_feat_names


def compute_spect_edge_freq(sfreq, data, ref_freq=None, edge=None,
                            psd_method='welch', psd_params=None):
    """Spectal Edge Frequency (per channel).

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    data : ndarray, shape (n_channels, n_times)

    ref_freq : float or None (default: None)
        If not None, reference frequency for the computation of the spectral
        edge frequency. If None, `ref_freq = sfreq / 2` is used.

    edge : list of float or None (default: None)
        If not None, ``edge`` is expected to be a list of values between 0
        and 1. If None, ``edge = [0.5]`` is used.

    psd_method : str (default: 'welch')
        Method used for the estimation of the Power Spectral Density (PSD).
        Valid methods are: ``'welch'``, ``'multitaper'`` or ``'fft'``.

    psd_params : dict or None (default: None)
        If not None, dict with optional parameters (`welch_n_fft`,
        `welch_n_per_seg`, `welch_n_overlap`) to be passed to
        :func:`mne_features.utils.power_spectrum`. If None, default parameters
        are used (see doc for :func:`mne_features.utils.power_spectrum`).

    Returns
    -------
    output : ndarray, shape (n_channels * n_edge,)
        With: `n_edge = 1` if `edge` is None or `n_edge = len(edge)` otherwise.

    Notes
    -----
    Alias of the feature function: **spect_edge_freq**. See [1]_.

    References
    ----------
    .. [1] Mormann, F. et al. (2006). Seizure prediction: the long and winding
           road. Brain, 130(2), 314-333.
    """
    if ref_freq is None:
        _ref_freq = sfreq / 2
    else:
        _ref_freq = float(ref_freq)
    if edge is None:
        _edge = [0.5]
    else:
        # Check the values in `edge`
        if not all([0 <= p <= 1 for p in edge]):
            raise ValueError('The values in ``edge``` must be floats between '
                             '0 and 1. Got {} instead.'.format(edge))
        else:
            _edge = edge
    n_edge = len(_edge)
    n_channels = data.shape[0]
    spect_edge_freq = np.empty((n_channels, n_edge))
    _psd_params = _psd_params_checker(psd_params)
    psd, freqs = power_spectrum(sfreq, data, psd_method=psd_method,
                                **_psd_params)
    out = np.cumsum(psd, 1)
    for i, p in enumerate(_edge):
        idx_ref = np.where(freqs >= _ref_freq)[0][0]
        ref_pow = np.sum(psd[:, :(idx_ref + 1)], axis=-1)
        for j in range(n_channels):
            idx = np.where(out[j, :] >= p * ref_pow[j])[0]
            if idx.size > 0:
                spect_edge_freq[j, i] = freqs[idx[0]]
            else:
                spect_edge_freq[j, i] = -1
    return spect_edge_freq.ravel()


def _compute_spect_edge_freq_feat_names(data, edge, **kwargs):
    """Utility function to create feature names compatible with the output of
    :func:`mne_features.univariate.compute_spect_edge_freq`."""
    n_channels = data.shape[0]
    if edge is None:
        _edge = [0.5]
    else:
        # Check the values in `edge`
        if not all([0 <= p <= 1 for p in edge]):
            raise ValueError('The values in ``edge``` must be floats between '
                             '0 and 1. Got {} instead.'.format(edge))
        else:
            _edge = edge
    n_edges = len(_edge)
    return ['ch%s_%s' % (ch, i) for ch in range(n_channels)
            for i in range(n_edges)]


compute_spect_edge_freq.get_feature_names = _compute_spect_edge_freq_feat_names


def compute_wavelet_coef_energy(data, wavelet_name='db4'):
    """Energy of Wavelet decomposition coefficients (per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    wavelet_name : str (default: db4)
        Wavelet name (to be used with ``pywt.Wavelet``). The full list of
        Wavelet names are given by: ``[name for family in pywt.families() for
        name in pywt.wavelist(family)]``.

    Returns
    -------
    output : ndarray, shape (n_channels * levdec,)
        The decomposition level (``levdec``) used for the DWT is either 6 or
        the maximum useful decomposition level (given the number of time points
        in the data and chosen wavelet ; see ``pywt.dwt_max_level``).

    Notes
    -----
    Alias of the feature function: **wavelet_coef_energy**. See [1]_.

    References
    ----------
    .. [1] Teixeira, C. A. et al. (2011). EPILAB: A software package for
           studies on the prediction of epileptic seizures. Journal of
           Neuroscience Methods, 200(2), 257-271.
    """
    n_channels = data.shape[0]
    coefs = _wavelet_coefs(data, wavelet_name)
    levdec = len(coefs) - 1
    wavelet_energy = np.zeros((n_channels, levdec))
    for j in range(n_channels):
        for level in range(levdec):
            wavelet_energy[j, level] = np.sum(coefs[levdec - level][j, :] ** 2)
    return wavelet_energy.ravel()


def _compute_wavelet_coef_energy_feat_names(data, wavelet_name, **kwargs):
    """Utility function to create feature names compatible with the output of
    :func:`mne_features.univariate.compute_wavelet_coef_energy`."""
    n_channels = data.shape[0]
    coefs = _wavelet_coefs(data, wavelet_name)
    levdec = len(coefs) - 1
    return ['ch%s_%s' % (ch, i) for ch in range(n_channels)
            for i in range(levdec)]


compute_wavelet_coef_energy.get_feature_names = \
    _compute_wavelet_coef_energy_feat_names


@nb.jit([nb.float64[:, :](nb.float64[:, :]),
         nb.float32[:, :](nb.float32[:, :])], nopython=True)
def _tk_energy(data):
    """Teager-Kaiser Energy.

    Utility function for :func:`compute_taeger_kaiser_energy`.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels, n_times - 2)
    """
    n_channels, n_times = data.shape
    tke = np.empty((n_channels, n_times - 2), dtype=data.dtype)
    for j in range(n_channels):
        for i in range(1, n_times - 1):
            tke[j, i - 1] = data[j, i] ** 2 - data[j, i - 1] * data[j, i + 1]
    return tke


def compute_teager_kaiser_energy(data, wavelet_name='db4'):
    """Compute the Teager-Kaiser energy.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    wavelet_name : str (default: 'db4')
        Wavelet name (to be used with ``pywt.Wavelet``). The full list of
        Wavelet names are given by: ``[name for family in pywt.families() for
        name in pywt.wavelist(family)]``.

    Returns
    -------
    output : ndarray, shape (n_channels * (levdec + 1) * 2,)

    Notes
    -----
    Alias of the feature function: **teager_kaiser_energy**. See [1]_.

    References
    ----------
    .. [1] Badani, S. et al. (2017). Detection of epilepsy based on discrete
           wavelet transform and Teager-Kaiser energy operator. In Calcutta
           Conference (CALCON). 2017 IEEE (pp. 164-167).
    """
    n_channels = data.shape[0]
    coefs = _wavelet_coefs(data, wavelet_name)
    levdec = len(coefs) - 1
    tke = np.empty((n_channels, levdec + 1, 2))
    for level in range(levdec + 1):
        tk_energy = _tk_energy(coefs[level])
        tke[:, level, 0] = np.mean(tk_energy, axis=-1)
        tke[:, level, 1] = np.std(tk_energy, ddof=1, axis=-1)
    return tke.ravel()


def _compute_teager_kaiser_energy_feat_names(data, wavelet_name, **kwargs):
    """Utility function to create feature names compatible with the output of
    :func:`mne_features.univariate.compute_teager_kaiser_energy`."""
    n_channels = data.shape[0]
    coefs = _wavelet_coefs(data, wavelet_name)
    levdec = len(coefs) - 1
    return ['ch%s_%s_%s' % (ch, i, stat) for ch in range(n_channels)
            for i in range(levdec + 1) for stat in ['mean', 'std']]


compute_teager_kaiser_energy.get_feature_names = \
    _compute_teager_kaiser_energy_feat_names
