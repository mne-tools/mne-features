# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


import numpy as np
from math import sqrt
from numpy.testing import assert_equal, assert_almost_equal, assert_raises

from mne_features.univariate import (_slope_lstsq, compute_mean,
                                     compute_variance, compute_std,
                                     compute_ptp_amp, compute_skewness,
                                     compute_kurtosis, compute_hurst_exp,
                                     compute_app_entropy, compute_samp_entropy,
                                     compute_decorr_time,
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
                                     compute_teager_kaiser_energy)

rng = np.random.RandomState(42)
sfreq = 256.
data1 = np.array([[0., 0., 2., -2., 0., -1., -1., 0.],
                  [1., 1., -1., -1., 0., 1., 1., 0.]])
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


def test_mean():
    expected = np.array([-1 / 4, 1 / 4])
    assert_almost_equal(compute_mean(data1), expected)


def test_variance():
    expected = np.array([19 / 14, 11 / 14])
    assert_almost_equal(compute_variance(data1), expected)


def test_std():
    expected = np.array([sqrt(19 / 14), sqrt(11 / 14)])
    assert_almost_equal(compute_std(data1), expected)


def test_skewness():
    expected = np.array([42 / (19 * sqrt(19)), -18 / (11 * sqrt(11))])
    assert_almost_equal(compute_skewness(data1), expected)


def test_kurtosis():
    expected = np.array([1141 / 361, 197 / 121])
    assert_almost_equal(compute_kurtosis(data1), expected)


def test_ptp_amp():
    expected = np.array([4, 2])
    assert_almost_equal(compute_ptp_amp(data1), expected)


def test_line_length():
    expected = np.array([10 / 7, 5 / 7])
    assert_almost_equal(compute_line_length(data1), expected)


def test_zero_crossings():
    expected = np.array([5, 3])
    assert_almost_equal(compute_zero_crossings(data1), expected)


def test_shape_output():
    for func in (compute_hurst_exp, compute_hjorth_complexity,
                 compute_hjorth_mobility, compute_higuchi_fd, compute_katz_fd,
                 compute_svd_entropy, compute_svd_fisher_info):
        for j in range(n_epochs):
            feat = func(data[j, :, :])
            assert_equal(feat.shape, (n_channels,))


def test_shape_output_decorr_time():
    for j in range(n_epochs):
        feat = compute_decorr_time(sfreq, data[j, :, :])
        assert_equal(feat.shape, (n_channels,))


def test_shape_output_pow_freq_bands():
    fb = np.array([0.1, 4, 8, 12, 30])
    n_freqs = fb.shape[0]
    for j in range(n_epochs):
        feat = compute_pow_freq_bands(sfreq, data[j, :, :], freq_bands=fb)
        assert_equal(feat.shape, (n_channels * (n_freqs - 1),))


def test_shape_output_hjorth_mobility_spect():
    for j in range(n_epochs):
        feat = compute_hjorth_mobility_spect(sfreq, data[j, :, :])
        assert_equal(feat.shape, (n_channels,))


def test_shape_output_hjorth_complexity_spect():
    for j in range(n_epochs):
        feat = compute_hjorth_complexity_spect(sfreq, data[j, :, :])
        assert_equal(feat.shape, (n_channels,))


def test_shape_output_spect_entropy():
    for j in range(n_epochs):
        feat = compute_spect_entropy(sfreq, data[j, :, :])
        assert_equal(feat.shape, (n_channels,))


def test_shape_output_energy_freq_bands():
    fb = np.array([0.1, 4, 8, 12, 30])
    n_freqs = fb.shape[0]
    for j in range(n_epochs):
        feat = compute_energy_freq_bands(sfreq, data[j, :, :], freq_bands=fb)
        assert_equal(feat.shape, (n_channels * (n_freqs - 1),))


def test_shape_output_spect_edge_freq():
    edge = [50., 80., 85., 95.]
    for j in range(n_epochs):
        feat = compute_spect_edge_freq(sfreq, data[j, :, :], edge=edge)
        assert_equal(feat.shape, (n_channels * 4,))


def test_shape_output_wavelet_coef_energy():
    feat = compute_wavelet_coef_energy(data[0, :, :], wavelet_name='haar')
    assert_equal(feat.shape, (n_channels * 6,))


def test_app_entropy():
    feat = compute_app_entropy(data[0, :, :], emb=5)
    assert_equal(feat.shape, (n_channels,))
    with assert_raises(ValueError):
        compute_app_entropy(data[0, :, :], emb=5, metric='sqeuclidean')


def test_samp_entropy():
    feat = compute_samp_entropy(data[0, :, :], emb=5)
    assert_equal(feat.shape, (n_channels,))
    with assert_raises(ValueError):
        compute_samp_entropy(data[0, :, :], emb=5, metric='sqeuclidean')


def test_shape_output_teager_kaiser_energy():
    feat = compute_teager_kaiser_energy(data[0, :, :])
    assert_equal(feat.shape, (n_channels * 6 * 2, ))


if __name__ == '__main__':

    test_slope_lstsq()
    test_mean()
    test_variance()
    test_std()
    test_skewness()
    test_kurtosis()
    test_ptp_amp()
    test_line_length()
    test_zero_crossings()
    test_shape_output()
    test_shape_output_decorr_time()
    test_shape_output_pow_freq_bands()
    test_shape_output_spect_entropy()
    test_shape_output_energy_freq_bands()
    test_shape_output_spect_edge_freq()
    test_shape_output_wavelet_coef_energy()
    test_app_entropy()
    test_samp_entropy()
    test_shape_output_teager_kaiser_energy()
