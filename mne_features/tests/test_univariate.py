# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


from math import sqrt, log, cos

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_raises
import pywt

from mne_features.feature_extraction import extract_features
from mne_features.univariate import (_slope_lstsq, _accumulate_std,
                                     _accumulate_min, _accumulate_max,
                                     _freq_bands_helper,
                                     compute_mean, compute_variance,
                                     compute_std, compute_ptp_amp,
                                     compute_skewness, compute_kurtosis,
                                     compute_rms, compute_quantile,
                                     compute_hurst_exp, compute_app_entropy,
                                     compute_samp_entropy, compute_decorr_time,
                                     compute_pow_freq_bands,
                                     compute_hjorth_mobility_spect,
                                     compute_hjorth_complexity_spect,
                                     compute_hjorth_mobility,
                                     compute_hjorth_complexity,
                                     compute_higuchi_fd, compute_katz_fd,
                                     compute_zero_crossings,
                                     compute_line_length,
                                     compute_spect_entropy,
                                     compute_svd_entropy,
                                     compute_svd_fisher_info,
                                     compute_energy_freq_bands,
                                     compute_spect_edge_freq,
                                     compute_wavelet_coef_energy,
                                     compute_teager_kaiser_energy,
                                     compute_spect_slope)

rng = np.random.RandomState(42)
sfreq = 512.
data1 = np.array([[0., 0., 2., -2., 0., -1., -1., 0.],
                  [1., 1., -1., -1., 0., 1., 1., 0.]])
data2 = rng.standard_normal((20, int(sfreq)))
_tp = 2 * np.pi * np.arange(int(sfreq)) / sfreq
_data_sin = 0.1 * np.sin(5 * _tp) + 0.05 * np.sin(33 * _tp)
data_sin = _data_sin[None, :]
data = rng.standard_normal((10, 20, int(sfreq)))
n_epochs, n_channels = data.shape[:2]


def test_slope_lstsq():
    x = rng.standard_normal((100,))
    m = rng.uniform()
    y = m * x + 1
    s1 = _slope_lstsq(x, y)
    s2 = np.polyfit(x, y, 1)[0]
    assert_almost_equal(s1, m)
    assert_almost_equal(s1, s2)


def test_accumulate_max():
    expected = np.array([0, 0, 2, 2, 2, 2, 2, 2])
    assert_equal(_accumulate_max(data1[0, :]), expected)


def test_accumulate_min():
    expected = np.array([0, 0, 0, -2, -2, -2, -2, -2])
    assert_equal(_accumulate_min(data1[0, :]), expected)


def test_accumulate_std():
    expected = np.array([0, 0, sqrt(4. / 3), sqrt(8. / 3), sqrt(2),
                         sqrt(53. / 30), sqrt(11. / 7), sqrt(19. / 14)])
    assert_almost_equal(_accumulate_std(data1[0, :]), expected)


def test_mean():
    expected = np.array([-1. / 4, 1. / 4])
    assert_almost_equal(compute_mean(data1), expected)


def test_variance():
    expected = np.array([19. / 14, 11. / 14])
    assert_almost_equal(compute_variance(data1), expected)


def test_std():
    expected = np.array([sqrt(19. / 14), sqrt(11. / 14)])
    assert_almost_equal(compute_std(data1), expected)


def test_skewness():
    expected = np.array([42. / (19 * sqrt(19)), -18. / (11 * sqrt(11))])
    assert_almost_equal(compute_skewness(data1), expected)


def test_kurtosis():
    expected = np.array([1141. / 361, 197. / 121])
    assert_almost_equal(compute_kurtosis(data1), expected)


def test_ptp_amp():
    expected = np.array([4, 2])
    assert_almost_equal(compute_ptp_amp(data1), expected)


def test_rms():
    expected = np.array([sqrt(1.25), sqrt(0.75)])
    assert_almost_equal(compute_rms(data1), expected)


def test_quantile():
    expected = np.array([0., 1.])
    assert_almost_equal(compute_quantile(data1, q=0.75), expected)
    # Multiple quantiles
    expected = np.array([-1, 0, -0.25, 1.])
    assert_almost_equal(compute_quantile(data1, q=[0.25, 0.75]), expected)


def test_line_length():
    expected = np.array([10. / 7, 5. / 7])
    assert_almost_equal(compute_line_length(data1), expected)


def test_zero_crossings():
    expected = np.array([4, 3])
    assert_almost_equal(compute_zero_crossings(data1), expected)
    _data = np.sin(np.linspace(0, 4 * 2 * np.pi, 20))
    assert_equal(compute_zero_crossings(_data[None, :], 1e-12), np.array([9]))
    assert_equal(compute_zero_crossings(_data[None, :]), np.array([8]))
    _data = np.array([0, 0, 0, 0, 0, 0, 0.1])
    assert_equal(compute_zero_crossings(_data[None, :]), np.array([1]))


def test_hurst_exp():
    """Test for :func:`compute_hurst_exp`.

    According to [Felle51]_ and [Anis76]_, the Hurst exponent for white noise
    is asymptotically equal to 0.5.

    References
    ----------
    .. [Felle51] Feller, W. (1951). The asymptotic distribution of the range
                 of sums of independent random variables. The Annals of
                 Mathematical Statistics, 427-432.

    .. [Anis76] Annis, A. A., & Lloyd, E. H. (1976). The expected value of the
                adjusted rescaled Hurst range of independent normal summands.
                Biometrika, 63(1), 111-116.
    """
    expected = 0.5 * np.ones((n_channels,))
    assert_almost_equal(compute_hurst_exp(data2), expected, decimal=1)


def test_app_entropy():
    expected = np.array([-log(7) + log(6),
                         (2 * log(2) - 7 * log(7)) / 7 + log(6)])
    assert_almost_equal(compute_app_entropy(data1), expected)
    # Note: the approximate entropy should be close to 0 for a
    # regular and predictable time series.
    data3 = np.array([(-1) ** np.arange(int(sfreq))])
    assert_almost_equal(compute_app_entropy(data3), 0, decimal=5)
    # Wrong `metric` parameter:
    with assert_raises(ValueError):
        compute_app_entropy(data[0, :, :], emb=5, metric='sqeuclidean')


def test_samp_entropy():
    _data = np.array([[1, -1, 1, -1, 0, 1, -1, 1]])
    expected = np.array([log(3)])
    assert_almost_equal(compute_samp_entropy(_data), expected)
    with assert_raises(ValueError):
        # Data for which SampEn is not defined:
        compute_samp_entropy(data1)
        # Wrong `metric` parameter:
        compute_samp_entropy(_data, metric='sqeuclidean')


def test_decorr_time():
    output = compute_decorr_time(sfreq, data2)
    # Output shape:
    assert_equal(output.shape, (n_channels,))
    # Decorrelation times should all be > 0
    assert_equal(np.all(compute_decorr_time(sfreq, data2) > 0), True)


def test_freq_bands_helper():
    fb1 = np.array([.5, 4, 8, 13, 30, 100])
    fb2 = np.array([[.5, 4], [4, 8], [8, 13], [13, 30], [30, 100]])
    assert_equal(fb2, _freq_bands_helper(256., fb1))
    assert_equal(fb2, _freq_bands_helper(256., fb2))
    with assert_raises(ValueError):
        _freq_bands_helper(128., fb1)
    with assert_raises(ValueError):
        _freq_bands_helper(256., fb2.T)


def test_pow_freq_bands_norm():
    expected = np.array([0, 0.005, 0, 0, 0.00125]) / 0.00625
    assert_almost_equal(
        compute_pow_freq_bands(sfreq, data_sin, normalize=True,
                               psd_method='fft'), expected)


def test_pow_freq_bands_no_norm():
    expected = np.array([0, 0.005, 0, 0, 0.00125])
    assert_almost_equal(
        compute_pow_freq_bands(sfreq, data_sin, normalize=False,
                               psd_method='fft'), expected)


def test_pow_freq_bands_norm_log():
    expected = np.array([-5317.19500282, -368.16479931, -5247.13240494,
                         -5146.55896452, -464.49439792])
    assert_almost_equal(
        compute_pow_freq_bands(sfreq, data_sin, normalize=True, log=True,
                               psd_method='fft'), expected)


def test_pow_freq_bands_no_norm_log():
    expected = np.array([-33.23246877, -2.30103, -32.79457753, -32.16599353,
                         -2.90308999])
    assert_almost_equal(
        compute_pow_freq_bands(sfreq, data_sin, normalize=False, log=True,
                               psd_method='fft'), expected)


def test_pow_freq_bands_ratios():
    # Ratios of power in bands:
    # For data_sin, only the usual theta (4Hz - 8Hz) and low gamma
    # (30Hz - 70Hz) bands contain non-zero power.
    fb = np.array([[4., 8.], [30., 70.]])
    expected_pow = np.array([0.005, 0.00125]) / 0.00625
    expected_ratios = np.array([4., 0.25])
    assert_almost_equal(
        compute_pow_freq_bands(
            sfreq, data_sin, freq_bands=fb, ratios='all', psd_method='fft'),
        np.r_[expected_pow, expected_ratios])
    assert_almost_equal(
        compute_pow_freq_bands(
            sfreq, data_sin, freq_bands=fb, ratios='only', psd_method='fft'),
        expected_ratios)
    assert_almost_equal(
        compute_pow_freq_bands(
            sfreq, data_sin, freq_bands=fb, ratios='only', ratios_triu=True,
            psd_method='fft'), expected_ratios[[0]])

    with assert_raises(ValueError):
        # Invalid `ratios` parameter
        compute_pow_freq_bands(sfreq, data_sin, ratios=['alpha', 'beta'])


def test_pow_freq_bands_log_ratios():
    # Log ratios of power in bands:
    # For data_sin, only the usual theta (4Hz - 8Hz) and low gamma
    # (30Hz - 70Hz) bands contain non-zero power.
    fb = np.array([[4., 8.], [30., 70.]])
    expected_pow = np.log10(np.array([0.005, 0.00125]))
    expected_ratios = np.log10(np.array([4., 0.25]))
    assert_almost_equal(
        compute_pow_freq_bands(
            sfreq, data_sin, freq_bands=fb, normalize=False, ratios='all',
            psd_method='fft', log=True), np.r_[expected_pow, expected_ratios])
    assert_almost_equal(
        compute_pow_freq_bands(
            sfreq, data_sin, freq_bands=fb, normalize=False, ratios='only',
            psd_method='fft', log=True), expected_ratios)


def test_generic_features_names():
    n_chans = 2  # keep only 2 channels for the sake of simplicity
    _data = data[:, :n_chans, :]
    selected_funcs = (
        ['mean', 'variance', 'std', 'ptp_amp', 'skewness', 'kurtosis', 'rms',
         'quantile', 'hurst_exp', 'app_entropy', 'samp_entropy', 'decorr_time',
         'hjorth_mobility_spect', 'hjorth_complexity_spect', 'hjorth_mobility',
         'hjorth_complexity', 'higuchi_fd', 'katz_fd', 'zero_crossings',
         'line_length', 'spect_entropy', 'svd_entropy', 'svd_fisher_info']
    )

    col_names = [(func, 'ch%s' % ch)
                 for func in selected_funcs for ch in range(n_chans)]
    df = extract_features(_data, sfreq, selected_funcs, return_as_df=True)
    assert df.columns.to_list() == col_names


def test_feature_names_quantile():
    n_chans = 2  # keep only 2 channels for the sake of simplicity
    _data = data[:, :n_chans, :]
    selected_funcs = ['quantile']

    q = [0.25, 0.75]
    col_names = ['ch%s_%s' % (ch, i) for ch in range(n_chans)
                 for i in range(len(q))]
    df = extract_features(
        _data, sfreq, selected_funcs, funcs_params={'quantile__q': q},
        return_as_df=True)
    assert_equal(df.columns.get_level_values(1).values, col_names)


def test_feature_names_spect_edge_freq():
    n_chans = 2  # keep only 2 channels for the sake of simplicity
    _data = data[:, :n_chans, :]
    selected_funcs = ['spect_edge_freq']

    _edges = [None, [.5], [.5, .9]]

    for edge in _edges:
        if edge is None:
            edge = [.5]
        col_names = ['ch%s_%s' % (ch, i) for ch in range(n_chans) for
                     i in range(len(edge))]
        df = extract_features(
            _data, sfreq, selected_funcs,
            funcs_params={'spect_edge_freq__edge': edge},
            return_as_df=True)
        assert_equal(df.columns.get_level_values(1).values, col_names)


def test_feature_names_spect_slope():
    n_chans = 2  # keep only 2 channels for the sake of simplicity
    _data = data[:, :n_chans, :]
    selected_funcs = ['spect_slope']

    stats = ['intercept', 'slope', 'MSE', 'R2']

    col_names = ['ch%s_%s' % (ch, stat) for ch in range(n_chans) for
                 stat in stats]
    df = extract_features(_data, sfreq, selected_funcs, return_as_df=True)
    assert_equal(df.columns.get_level_values(1).values, col_names)


def test_feature_names_wavelet_coef_energy(wavelet_name='db4'):
    n_chans = 2  # keep only 2 channels for the sake of simplicity
    _data = data[:, :n_chans, :]
    selected_funcs = ['wavelet_coef_energy']

    # number of coefficients of the DWT
    wavelet = pywt.Wavelet(wavelet_name)
    levdec = min(pywt.dwt_max_level(_data.shape[-1], wavelet.dec_len), 6)

    col_names = ['ch%s_%s' % (ch, i) for ch in range(n_chans) for
                 i in range(levdec)]

    df = extract_features(
            _data, sfreq, selected_funcs,
            funcs_params={'wavelet_coef_energy__wavelet_name': wavelet_name},
            return_as_df=True)
    assert_equal(df.columns.get_level_values(1).values, col_names)


def test_feature_names_teager_kaiser_energy(wavelet_name='db4'):
    n_chans = 2  # keep only 2 channels for the sake of simplicity
    _data = data[:, :n_chans, :]
    selected_funcs = ['teager_kaiser_energy']

    # number of coefficients of the DWT
    wavelet = pywt.Wavelet(wavelet_name)
    levdec = min(pywt.dwt_max_level(_data.shape[-1], wavelet.dec_len), 6)

    col_names = ['ch%s_%s_%s' % (ch, i, stat) for ch in range(n_chans)
                 for i in range(levdec + 1) for stat in ['mean', 'std']]

    df = extract_features(
            _data, sfreq, selected_funcs,
            funcs_params={'teager_kaiser_energy__wavelet_name': wavelet_name},
            return_as_df=True)
    assert_equal(df.columns.get_level_values(1).values, col_names)


def test_feature_names_pow_freq_bands():
    _data = data[:, :2, :]  # keep only 2 channels for the sake of simplicity
    selected_funcs = ['pow_freq_bands']
    fb1 = np.array([[4., 8.], [30., 70.]])
    fb2 = {'theta': [4, 8], 'low-gamma': np.array([30, 70])}
    _fb = [fb1, fb2]
    ratios_col_names1 = ['ch0_band0/band1', 'ch0_band1/band0',
                         'ch1_band0/band1', 'ch1_band1/band0']
    ratios_col_names2 = ['ch0_theta/low-gamma', 'ch0_low-gamma/theta',
                         'ch1_theta/low-gamma', 'ch1_low-gamma/theta']
    _ratios_names = [ratios_col_names1, ratios_col_names2]
    pow_col_names1 = ['ch0_band0', 'ch0_band1', 'ch1_band0', 'ch1_band1']
    pow_col_names2 = ['ch0_theta', 'ch0_low-gamma',
                      'ch1_theta', 'ch1_low-gamma']
    _pow_names = [pow_col_names1, pow_col_names2]

    for fb, ratios_names, pow_names in zip(_fb, _ratios_names, _pow_names):
        # With `ratios = 'only'`:
        df_only = extract_features(
            _data, sfreq, selected_funcs,
            funcs_params={'pow_freq_bands__ratios': 'only',
                          'pow_freq_bands__freq_bands': fb},
            return_as_df=True)
        assert_equal(df_only.columns.get_level_values(1).values, ratios_names)

        # With `ratios = 'all'`:
        df_all = extract_features(
            _data, sfreq, selected_funcs,
            funcs_params={'pow_freq_bands__ratios': 'all',
                          'pow_freq_bands__freq_bands': fb},
            return_as_df=True)
        assert_equal(df_all.columns.get_level_values(1).values,
                     pow_names + ratios_names)

        # With `ratios = None`:
        df = extract_features(
            _data, sfreq, selected_funcs,
            funcs_params={'pow_freq_bands__ratios': None,
                          'pow_freq_bands__freq_bands': fb},
            return_as_df=True)
        assert_equal(df.columns.get_level_values(1).values, pow_names)

        # With `ratios = 'only'` and `ratios_triu = True`:
        df_only = extract_features(
            _data, sfreq, selected_funcs,
            funcs_params={'pow_freq_bands__ratios': 'only',
                          'pow_freq_bands__ratios_triu': True,
                          'pow_freq_bands__freq_bands': fb},
            return_as_df=True)
        assert_equal(df_only.columns.get_level_values(1).values,
                     ratios_names[::2])


def test_hjorth_mobility_spect():
    expected = 0.005 * (5 ** 2) + 0.00125 * (33 ** 2)
    assert_almost_equal(compute_hjorth_mobility_spect(sfreq, data_sin,
                                                      psd_method='fft'),
                        expected)


def test_hjorth_complexity_spect():
    expected = 0.005 * (5 ** 4) + 0.00125 * (33 ** 4)
    assert_almost_equal(compute_hjorth_complexity_spect(sfreq, data_sin,
                                                        psd_method='fft'),
                        expected)
    assert_almost_equal(compute_hjorth_complexity_spect(sfreq, data_sin,
                                                        normalize=True,
                                                        psd_method='fft'),
                        expected / 0.00625)


def test_hjorth_mobility():
    expected = np.array([(6 * sqrt(26)) / (sqrt(7) * sqrt(43)),
                         (6 * sqrt(8)) / (5 * sqrt(7))])
    assert_almost_equal(compute_hjorth_mobility(data1), expected)


def test_hjorth_complexity():
    expected = np.array([sqrt(29885) / 156, (5 * sqrt(103)) / 48])
    compute_hjorth_complexity(data1)
    assert_almost_equal(compute_hjorth_complexity(data1), expected)


def test_higuchi_fd():
    """Test for :func:`compute_higuchi_fd`.

    According to [Este01a]_, the Weierstrass Cosine Function (WCF) with
    parameter H (such that 0 < H < 1) can be used to produce a signal whose
    fractal dimension equals 2 - H.

    References
    ----------
    .. [Este01a] Esteller, R. et al. (2001). A comparison of waveform fractal
                 dimension algorithms. IEEE Transactions on Circuits and
                 Systems I: Fundamental Theory and Applications, 48(2),
                 177-183.
    """
    t = np.linspace(0, 1, 1024)
    _wcf = np.empty((1024,))
    H = 0.5  # WCF parameter
    for j in range(1024):
        _wcf[j] = sum([(5 ** (-H * i) * cos(2 * np.pi * (5 ** i) * t[j]))
                       for i in range(26)])
    wcf = np.array(_wcf)[None, :]
    assert_almost_equal(compute_higuchi_fd(wcf), 2 - H, decimal=1)


def test_katz_fd():
    """Test for :func:`compute_katz_fd`.

    As discussed in [Este01a]_, Katz fractal dimension is not as accurate as
    Higuchi fractal dimension when tested on synthetic data (see
    :func:`test_higuchi_fd`).
    """
    expected = np.array([log(7, 10.) / (log(2. / 10, 10.) + log(7, 10.)),
                         log(7, 10.) / (log(2. / 5, 10.) + log(7, 10.))])
    assert_almost_equal(compute_katz_fd(data1), expected)


def test_energy_freq_bands():
    """Test for :func:`compute_energy_freq_bands`.

    For `data_sin` (signal x(t) = 0.1 * sin(5t) + 0.05 * sin(33t), on
    [0, 2 * pi], at sfreq = 512Hz), the 512-points FFT is everywhere zero,
    except for the bins corresponding to frequencies +/-5Hz and +/-33Hz.
    Therefore, all the power/energy of the signal should be in the [1Hz - 40Hz]
    frequency band. As a result, this test passes if more than 98% of the
    energy of the signal lies in the [1Hz - 40Hz] band.
    """
    band_energy = compute_energy_freq_bands(sfreq, data_sin,
                                            freq_bands=np.array([1., 40.]),
                                            deriv_filt=False)
    tot_energy = np.sum(data_sin ** 2, axis=-1)
    assert_equal(band_energy > 0.98 * tot_energy, True)


def test_feature_names_energy_freq_bands():
    _data = data[:, :2, :]  # keep only 2 channels for the sake of simplicity
    selected_funcs = ['energy_freq_bands']
    fb1 = np.array([[4., 8.], [30., 70.]])
    fb2 = {'theta': [4, 8], 'low-gamma': np.array([30, 70])}
    _fb = [fb1, fb2]
    expected_names1 = ['ch0_band0', 'ch0_band1', 'ch1_band0', 'ch1_band1']
    expected_names2 = ['ch0_theta', 'ch0_low-gamma',
                       'ch1_theta', 'ch1_low-gamma']
    _expected_names = [expected_names1, expected_names2]

    for fb, feat_names in zip(_fb, _expected_names):

        df = extract_features(
            _data, sfreq, selected_funcs,
            funcs_params={'energy_freq_bands__freq_bands': fb},
            return_as_df=True)
        assert_equal(df.columns.get_level_values(1).values, feat_names)


def test_spect_slope():
    """Test for :func:`compute_powercurve_deviation`.

    We impose the power to be written as power(f) = k1/f**theta with noise
    and derive a signal candidate, to which we apply the function
    compute_powercurve_deviation and check whether the k1 and theta estimates
    are correct.
    """
    # Support of the spectrum
    freqs = np.fft.fftfreq(n=int(sfreq), d=1. / sfreq)

    # parameters
    k1 = 5.
    theta = 3.

    # Define the magnitude of the spectrum
    # such that power(f) = k1/f**a with noise
    mag = np.zeros((freqs.shape[0],))
    mag[0] = 0
    noise = rng.uniform(low=-0.01, high=0.01, size=127)
    pos_freqs = np.arange(1, 128)
    mag[pos_freqs] = (np.sqrt(k1) + noise) / (pos_freqs ** (theta / 2))
    mag[-pos_freqs] = mag[1:128]

    # From the magnitude, we get the spectrum, choosing a random phase per bin.
    spect = np.zeros((freqs.shape[0],), dtype=np.complex64)
    spect[0] = 0
    phase = rng.uniform(low=-np.pi, high=np.pi, size=127)
    spect[pos_freqs] = mag[pos_freqs] * np.exp(phase * 1j)
    spect[-pos_freqs] = np.conj(spect[pos_freqs])

    # Take the inverse FFT to go back to the time domain.
    # The imaginary part of _sig is numerically close to 0
    # by hermitian symmetry. So we obtain a real signal.
    _sig = np.fft.ifft(spect)
    sig = _sig.real
    n_times = sig.shape[0]

    # We test our estimates
    intercept, slope, mse, r2 = \
        compute_spect_slope(sfreq=sfreq, data=sig.reshape(1, -1),
                            with_intercept=True, psd_method='fft')

    # obtained by the expression ps[f] = 2 * [ (spect[f]^2) / (n_times^2) ]
    # and plug-in: power(f) = k1/f**theta with noise
    k1_estimate = 10**(intercept - np.log10(2) + 2 * np.log10(n_times))
    theta_estimate = - slope

    np.testing.assert_almost_equal(k1, k1_estimate, decimal=1)
    np.testing.assert_almost_equal(theta, theta_estimate, decimal=1)
    assert r2 > 0.95, "Explained variance is not high enough."
    assert mse < 0.5, "Residual has too large standard deviation."


def test_spect_entropy():
    expected = -(0.005 / 0.00625) * log(0.005 / 0.00625, 2.) - \
        (0.00125 / 0.00625) * log(0.00125 / 0.00625, 2.)
    assert_almost_equal(compute_spect_entropy(sfreq, data_sin,
                                              psd_method='fft'), expected)


def test_spect_edge_freq():
    """Test for :func:`compute_spect_edge_freq`.

    For `data_sin` (signal x(t) = 0.1 * sin(5t) + 0.05 * sin(33t), on
    [0, 2 * pi], at sfreq = 512Hz), the minimum frequency at which more than
    50% (resp. 90%) of the spectral power up to ref_freq = 15Hz
    (resp. ref_freq = 50Hz) is contained in the signal is 5Hz (resp. 33HZ).
    """
    expected = 5.
    assert_almost_equal(compute_spect_edge_freq(sfreq, data_sin, ref_freq=15,
                                                edge=[.5], psd_method='fft'),
                        expected)
    expected = 33.
    assert_almost_equal(compute_spect_edge_freq(sfreq, data_sin, ref_freq=50,
                                                edge=[.9], psd_method='fft'),
                        expected)


def test_svd_entropy():
    assert_equal(compute_svd_entropy(data2, tau=2, emb=2) > 0, True)


def test_svd_fisher_info():
    assert_equal(compute_svd_fisher_info(data2, tau=2, emb=2) > 0, True)


def test_shape_output_wavelet_coef_energy():
    feat = compute_wavelet_coef_energy(data[0, :, :], wavelet_name='haar')
    assert_equal(feat.shape, (n_channels * 6,))


def test_shape_output_teager_kaiser_energy():
    feat = compute_teager_kaiser_energy(data[0, :, :])
    assert_equal(feat.shape, (n_channels * 7 * 2, ))


if __name__ == '__main__':

    test_slope_lstsq()
    test_accumulate_max()
    test_accumulate_min()
    test_accumulate_std()
    test_mean()
    test_variance()
    test_std()
    test_skewness()
    test_kurtosis()
    test_ptp_amp()
    test_rms()
    test_quantile()
    test_line_length()
    test_zero_crossings()
    test_hurst_exp()
    test_app_entropy()
    test_samp_entropy()
    test_decorr_time()
    test_freq_bands_helper()
    test_pow_freq_bands_norm()
    test_pow_freq_bands_no_norm()
    test_pow_freq_bands_norm_log()
    test_pow_freq_bands_no_norm_log()
    test_pow_freq_bands_ratios()
    test_pow_freq_bands_log_ratios()
    test_feature_names_pow_freq_bands()
    test_hjorth_mobility_spect()
    test_hjorth_complexity_spect()
    test_hjorth_mobility()
    test_hjorth_complexity()
    test_higuchi_fd()
    test_katz_fd()
    test_energy_freq_bands()
    test_feature_names_energy_freq_bands()
    test_spect_slope()
    test_spect_entropy()
    test_spect_edge_freq()
    test_svd_entropy()
    test_svd_fisher_info()
    test_shape_output_wavelet_coef_energy()
    test_shape_output_teager_kaiser_energy()
    test_generic_features_names()
    test_feature_names_spect_edge_freq()
    test_feature_names_spect_slope()
    test_feature_names_wavelet_coef_energy()
    test_feature_names_teager_kaiser_energy()
