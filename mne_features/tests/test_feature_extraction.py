# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


import numpy as np
from numpy.testing import assert_equal
from mne_features.feature_extraction import extract_features


def test_shape_output():
    rng = np.random.RandomState(42)
    X = rng.standard_normal((5, 10, 256))
    Xnew = extract_features(X, 256., ['mean', 'variance', 'kurtosis'])
    assert_equal(Xnew.shape, (5, 30))

if __name__ == '__main__':

    test_shape_output()
