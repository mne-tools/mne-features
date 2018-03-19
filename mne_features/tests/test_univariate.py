# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from mne_features.univariate import (_slope_lstsq, compute_mean,
                                     compute_variance, compute_std,
                                     compute_ptp, compute_skewness,
                                     compute_kurtosis, compute_hurst_exponent,
                                     compute_app_entropy, compute_samp_entropy,
                                     compute_decorr_time,
                                     compute_power_spectrum_freq_bands,
                                     compute_spect_hjorth_mobility,
                                     compute_spect_hjorth_complexity,
                                     compute_hjorth_mobility,
                                     compute_hjorth_complexity,
                                     compute_higuchi_fd, compute_katz_fd,
                                     compute_zero_crossings,
                                     compute_line_length,
                                     compute_spect_entropy,
                                     compute_svd_entropy,
                                     compute_svd_fisher_info,
                                     compute_energy_freq_bands)

rng = np.random.RandomState(42)
sfreq = 256.
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


def test_shape_output():
    for func in (compute_mean, compute_variance, compute_std,
                 compute_kurtosis, compute_skewness, compute_ptp,
                 compute_hurst_exponent, compute_app_entropy,
                 compute_samp_entropy, compute_hjorth_complexity,
                 compute_hjorth_mobility, compute_higuchi_fd, compute_katz_fd,
                 compute_zero_crossings, compute_line_length,
                 compute_svd_entropy, compute_svd_fisher_info):
        for j in range(n_epochs):
            feat = func(data[j, :, :])
            assert_equal(feat.shape, (n_channels,))


def test_shape_output_decorr_time():
    for j in range(n_epochs):
        feat = compute_decorr_time(sfreq, data[j, :, :])
        assert_equal(feat.shape, (n_channels,))


def test_shape_output_power_spectrum_freq_bands():
    fb = np.array([0.1, 4, 8, 12, 30])
    n_freqs = fb.shape[0]
    for j in range(n_epochs):
        feat = compute_power_spectrum_freq_bands(sfreq, fb, data[j, :, :])
        assert_equal(feat.shape, (n_channels * (n_freqs - 1),))


def test_shape_output_spect_hjorth_mobility():
    for j in range(n_epochs):
        feat = compute_spect_hjorth_mobility(sfreq, data[j, :, :])
        assert_equal(feat.shape, (n_channels,))


def test_shape_output_spect_hjorth_complexity():
    for j in range(n_epochs):
        feat = compute_spect_hjorth_complexity(sfreq, data[j, :, :])
        assert_equal(feat.shape, (n_channels,))


def test_shape_output_spect_entropy():
    for j in range(n_epochs):
        feat = compute_spect_entropy(sfreq, data[j, :, :])
        assert_equal(feat.shape, (n_channels,))


def test_shape_output_energy_freq_bands():
    fb = np.array([0.1, 4, 8, 12, 30])
    n_freqs = fb.shape[0]
    for j in range(n_epochs):
        feat = compute_energy_freq_bands(sfreq, fb, data[j, :, :])
        assert_equal(feat.shape, (n_channels * (n_freqs - 1),))


if __name__ == '__main__':

    test_slope_lstsq()
    test_shape_output()
    test_shape_output_decorr_time()
    test_shape_output_power_spectrum_freq_bands()
    test_shape_output_spect_entropy()
    test_shape_output_energy_freq_bands()
