# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


import mne
import numpy as np
import os.path as op
from numpy.testing import assert_equal
from mne_features.bivariate import compute_max_cross_correlation


path_data = op.join(op.dirname(__file__), 'data', 'test_data_chbmit-epo.fif')
epochs = mne.read_epochs(path_data)
data = epochs.get_data()
n_epochs, n_channels = data.shape[:2]
sfreq = epochs.info['sfreq']
rng = np.random.RandomState(42)


def test_max_cross_corr():
    Xnew = compute_max_cross_correlation(sfreq, data[0, :, :])
    assert_equal(Xnew.shape, (n_channels * (n_channels + 1) / 2,))


if __name__ == '__main__':

    test_max_cross_corr()
