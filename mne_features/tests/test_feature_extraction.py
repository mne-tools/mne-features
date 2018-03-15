# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


import numpy as np
from numpy.testing import assert_equal

from mne_features.feature_extraction import extract_features

rng = np.random.RandomState(42)
sfreq = 256.
data = rng.standard_normal((10, 20, int(sfreq)))
n_epochs, n_channels = data.shape[:2]


def test_shape_output_feature_extraction():
    freq_bands = np.array([0.1, 4, 8, 12, 30, 70])
    n_freqs = freq_bands.shape[0]
    sel_funcs = ['mean', 'variance', 'pow_freq_bands', 'kurtosis']
    features = extract_features(data, sfreq, freq_bands, sel_funcs, n_jobs=1)
    expected_shape = (n_epochs, n_channels * (2 + n_freqs))
    assert_equal(features.shape, expected_shape)


def test_njobs_feature_extraction():
    freq_bands = np.array([0.1, 4, 8, 12, 30, 70])
    n_freqs = freq_bands.shape[0]
    sel_funcs = ['pow_freq_bands']
    features = extract_features(data, sfreq, freq_bands, sel_funcs, n_jobs=-1)
    expected_shape = (n_epochs, n_channels * (n_freqs - 1))
    assert_equal(features.shape, expected_shape)


if __name__ == '__main__':

    test_shape_output_feature_extraction()
    test_njobs_feature_extraction()
