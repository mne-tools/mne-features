# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

import numpy as np
from numpy.testing import assert_equal
from mne_features.univariate import (compute_kurtosis, compute_mean,
                                     compute_ptp, compute_skewness,
                                     compute_variance)  # noqa


def test_univariate():
    N, C, T = 3, 5, 30
    rng = np.random.RandomState(42)
    data = rng.randn(N, C, T)
    for func in (compute_kurtosis, compute_mean,
                 compute_ptp, compute_skewness,
                 compute_variance):
        X = func(data)
        assert_equal(X.shape, (N, C))

if __name__ == '__main__':

    test_univariate()
