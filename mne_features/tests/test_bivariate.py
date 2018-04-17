# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


import numpy as np
from numpy.testing import assert_equal

from mne_features.bivariate import (compute_max_cross_corr,
                                    compute_nonlin_interdep,
                                    compute_phase_lock_val,
                                    compute_spect_corr,
                                    compute_time_corr)

rng = np.random.RandomState(42)
sfreq = 64.
data = rng.standard_normal((3, 5, int(sfreq)))
n_epochs, n_channels = data.shape[:2]


def test_shape_output_max_cross_corr():
    feat = compute_max_cross_corr(sfreq, data[0, :, :], include_diag=True)
    feat_nodiag = compute_max_cross_corr(sfreq, data[0, :, :],
                                         include_diag=False)
    n_coefs = (n_channels * (n_channels + 1)) // 2
    n_coefs2 = (n_channels * (n_channels - 1)) // 2
    assert_equal(feat.shape, (n_coefs,))
    assert_equal(feat_nodiag.shape, (n_coefs2,))


def test_shape_output_nonlin_interdep():
    feat = compute_nonlin_interdep(data[0, :, :], include_diag=True)
    feat_nodiag = compute_nonlin_interdep(data[0, :, :], include_diag=False)
    n_coefs = (n_channels * (n_channels + 1)) // 2
    assert_equal(feat.shape, (n_coefs,))
    assert_equal(feat_nodiag.shape, (n_coefs - n_channels,))


def test_shape_output_phase_lock_val():
    feat = compute_phase_lock_val(data[0, :, :], include_diag=True)
    feat_nodiag = compute_phase_lock_val(data[0, :, :], include_diag=False)
    n_coefs = (n_channels * (n_channels + 1)) // 2
    assert_equal(feat.shape, (n_coefs,))
    assert_equal(feat_nodiag.shape, (n_coefs - n_channels,))


def test_shape_output_spect_corr():
    feat_eig = compute_spect_corr(sfreq, data[0, :, :], with_eigenvalues=True,
                                  include_diag=True)
    feat_eig_nodiag = compute_spect_corr(sfreq, data[0, :, :],
                                         with_eigenvalues=True,
                                         include_diag=False)
    feat = compute_spect_corr(sfreq, data[0, :, :], with_eigenvalues=False,
                              include_diag=True)
    feat_nodiag = compute_spect_corr(sfreq, data[0, :, :],
                                     with_eigenvalues=False,
                                     include_diag=False)
    n_coefs = (n_channels * (n_channels + 1)) // 2
    assert_equal(feat_eig.shape, (n_coefs + n_channels,))
    assert_equal(feat_eig_nodiag.shape, (n_coefs,))
    assert_equal(feat.shape, (n_coefs,))
    assert_equal(feat_nodiag.shape, (n_coefs - n_channels,))


def test_shape_output_time_corr():
    feat_eig = compute_time_corr(data[0, :, :], with_eigenvalues=True,
                                 include_diag=True)
    feat_eig_nodiag = compute_time_corr(data[0, :, :],
                                        with_eigenvalues=True,
                                        include_diag=False)
    feat = compute_time_corr(data[0, :, :], with_eigenvalues=False,
                             include_diag=True)
    feat_nodiag = compute_time_corr(data[0, :, :], with_eigenvalues=False,
                                    include_diag=False)
    n_coefs = (n_channels * (n_channels + 1)) // 2
    assert_equal(feat_eig.shape, (n_coefs + n_channels,))
    assert_equal(feat_eig_nodiag.shape, (n_coefs,))
    assert_equal(feat.shape, (n_coefs,))
    assert_equal(feat_nodiag.shape, (n_coefs - n_channels,))


if __name__ == '__main__':

    test_shape_output_max_cross_corr()
    test_shape_output_nonlin_interdep()
    test_shape_output_phase_lock_val()
    test_shape_output_spect_corr()
    test_shape_output_time_corr()
