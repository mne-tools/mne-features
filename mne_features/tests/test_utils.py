# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipy import signal

from mne_features.utils import triu_idx, power_spectrum, embed

rng = np.random.RandomState(42)
sfreq = 256.
data = rng.standard_normal((20, int(sfreq)))


def test_power_spectrum():
    ps, freqs = power_spectrum(sfreq, data, return_db=False)
    assert_almost_equal(np.mean(data ** 2, axis=-1), np.sum(ps, axis=-1))


def test_psd():
    x0 = data - np.mean(data, axis=-1)[:, None]
    freqs, pxx = signal.welch(data, sfreq,
                              window=signal.get_window('boxcar',
                                                       data.shape[-1]),
                              return_onesided=True, scaling='spectrum')
    ps, freqs2 = power_spectrum(sfreq, x0, return_db=False)
    assert_almost_equal(freqs, freqs2)
    assert_almost_equal(10. * np.log10(pxx), 10. * np.log10(ps))


def test_triu_idx():
    n_channels = data.shape[0]
    idx0, idx1 = np.triu_indices(n_channels)
    triu_indices = np.array([np.arange(idx0.size), idx0, idx1])
    triu_indices2 = np.array(list(triu_idx(n_channels)))
    assert_almost_equal(triu_indices, triu_indices2.transpose())


def test_shape_output_embed():
    d, tau = 10, 10
    emb_data = embed(data[0, :], d=d, tau=tau)
    expected = (data.shape[-1] - 1 - (d - 1) * tau, d)
    assert_equal(emb_data.shape, expected)


if __name__ == '__main__':

    test_power_spectrum()
    test_psd()
    test_triu_idx()
    test_shape_output_embed()
