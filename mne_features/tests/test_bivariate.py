# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


import numpy as np
from numpy.testing import assert_equal
from mne_features.bivariate import compute_max_cross_correlation


def test_max_cross_corr():
    rng = np.random.RandomState(42)
    X = rng.standard_normal((10, 256))
    Xnew = compute_max_cross_correlation(256., X)
    assert_equal(Xnew.shape, (55,))

if __name__ == '__main__':

    test_max_cross_corr()
