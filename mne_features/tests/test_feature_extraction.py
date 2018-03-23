# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


import numpy as np
from numpy.testing import assert_equal, assert_raises

from mne_features.feature_extraction import extract_features

rng = np.random.RandomState(42)
sfreq = 256.
data = rng.standard_normal((10, 20, int(sfreq)))
n_epochs, n_channels = data.shape[:2]


def test_shape_output():
    sel_funcs = ['mean', 'variance', 'kurtosis', 'pow_freq_bands']
    features = extract_features(data, sfreq, sel_funcs, n_jobs=1)
    features_as_df = extract_features(data, sfreq, sel_funcs,
                                      n_jobs=1, return_as_df=True)
    expected_shape = (n_epochs, (3 + 5) * n_channels)
    assert_equal(features.shape, expected_shape)
    assert_equal(features, features_as_df.values)


def test_njobs():
    sel_funcs = ['app_entropy']
    features = extract_features(data, sfreq, sel_funcs, n_jobs=-1)
    expected_shape = (n_epochs, n_channels)
    assert_equal(features.shape, expected_shape)


def test_optional_params():
    features1 = extract_features(data, sfreq, ['spect_edge_freq'],
                                 {'spect_edge_freq__edge': [0.6]})
    features2 = extract_features(data, sfreq, ['spect_edge_freq'],
                                 {'spect_edge_freq__edge': [0.5, 0.95]})
    features3 = extract_features(data, sfreq, ['svd_fisher_info'],
                                 {'svd_fisher_info__tau': 5})
    assert_equal(features1.shape[-1], n_channels)
    assert_equal(features3.shape[-1], n_channels)
    assert_equal(features2.shape[-1], features1.shape[-1] * 2)


def test_optional_params_func_with_numba():
    sel_funcs = ['higuchi_fd']
    features1 = extract_features(data, sfreq, sel_funcs,
                                 {'higuchi_fd__kmax': 5})
    n_features1 = features1.shape[-1]
    assert_equal(n_features1, n_channels)


def test_wrong_params():
    with assert_raises(ValueError):
        # Negative sfreq
        extract_features(data, -0.1, ['mean'])
    with assert_raises(ValueError):
        # Unknown alias of feature function
        extract_features(data, sfreq, ['powfreqbands'])
    with assert_raises(ValueError):
        # No alias given
        extract_features(data, sfreq, list())
    with assert_raises(ValueError):
        # Passing optional arguments for with unknown alias
        extract_features(data, sfreq, ['higuchi_fd'],
                         {'higuch_fd__kmax': 3})


if __name__ == '__main__':

    test_shape_output()
    test_njobs()
    test_optional_params()
    test_optional_params_func_with_numba()
    test_wrong_params()
