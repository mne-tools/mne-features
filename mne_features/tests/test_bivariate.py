# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


import numpy as np
from numpy.testing import assert_equal

from mne_features.bivariate import (compute_max_cross_correlation,
                                    compute_nonlinear_interdep,
                                    compute_phase_locking_value,
                                    compute_spect_corr_coefs,
                                    compute_time_corr_coefs)

rng = np.random.RandomState(42)
sfreq = 256.
data = rng.standard_normal((10, 20, int(sfreq)))
n_epochs, n_channels = data.shape[:2]


def test_shape_output_max_cross_corr():
    feat = compute_max_cross_correlation(sfreq, data[0, :, :])
    n_coefs = (n_channels * (n_channels + 1)) // 2
    assert_equal(feat.shape, (n_coefs,))


def test_shape_output_nonlinear_interdep():
    feat = compute_nonlinear_interdep(data[0, :, :])
    n_coefs = (n_channels * (n_channels + 1)) // 2
    assert_equal(feat.shape, (n_coefs,))


def test_shape_output_plv():
    feat = compute_phase_locking_value(data[0, :, :])
    n_coefs = (n_channels * (n_channels + 1)) // 2
    assert_equal(feat.shape, (n_coefs,))


def test_shape_output_spect_corr():
    feat_eig = compute_spect_corr_coefs(sfreq, data[0, :, :],
                                        with_eigenvalues=True)
    feat = compute_spect_corr_coefs(sfreq, data[0, :, :],
                                    with_eigenvalues=False)
    n_coefs = (n_channels * (n_channels + 1)) // 2
    assert_equal(feat_eig.shape, (n_coefs + n_channels,))
    assert_equal(feat.shape, (n_coefs,))


def test_shape_output_time_corr():
    feat_eig = compute_time_corr_coefs(data[0, :, :], with_eigenvalues=True)
    feat = compute_time_corr_coefs(data[0, :, :], with_eigenvalues=False)
    n_coefs = (n_channels * (n_channels + 1)) // 2
    assert_equal(feat_eig.shape, (n_coefs + n_channels,))
    assert_equal(feat.shape, (n_coefs,))


if __name__ == '__main__':

    test_shape_output_max_cross_corr()
    test_shape_output_nonlinear_interdep()
    test_shape_output_plv()
    test_shape_output_spect_corr()
    test_shape_output_time_corr()
