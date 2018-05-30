# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipy import signal

from mne_features.utils import _idxiter, power_spectrum, _embed, _filt


rng = np.random.RandomState(42)
sfreq = 256.
data = rng.standard_normal((20, int(sfreq)))


def test_power_spectrum():
    ps, freqs = power_spectrum(sfreq, data, return_db=False)
    _data = data - np.mean(data, axis=-1)[:, None]
    assert_almost_equal(np.mean(_data ** 2, axis=-1), np.sum(ps, axis=-1))


def test_psd():
    n_times = data.shape[-1]
    freqs, pxx = signal.welch(data, sfreq,
                              window=signal.get_window('boxcar', n_times),
                              return_onesided=True, scaling='spectrum')
    ps, freqs2 = power_spectrum(sfreq, data, return_db=False)
    assert_almost_equal(freqs, freqs2)
    assert_almost_equal(pxx, ps)


def test_idxiter():
    n_channels = data.shape[0]
    # Upper-triangular part, including diag
    idx0, idx1 = np.triu_indices(n_channels)
    triu_indices = np.array([np.arange(idx0.size), idx0, idx1])
    triu_indices2 = np.array(list(_idxiter(n_channels, include_diag=True)))
    # Upper-triangular part, without diag
    idx2, idx3 = np.triu_indices(n_channels, 1)
    triu_indices_nodiag = np.array([np.arange(idx2.size), idx2, idx3])
    triu_indices2_nodiag = np.array(list(_idxiter(n_channels,
                                                  include_diag=False)))
    assert_almost_equal(triu_indices, triu_indices2.transpose())
    assert_almost_equal(triu_indices_nodiag, triu_indices2_nodiag.transpose())
    # Upper and lower-triangular parts, without diag
    expected = [(i, j) for _, (i, j) in
                enumerate(np.ndindex((n_channels, n_channels))) if i != j]
    assert_equal(np.array([(i, j) for _, i, j in _idxiter(n_channels,
                                                          triu=False)]),
                 expected)


def test_embed():
    d, tau = 10, 10
    emb_data = _embed(data, d=d, tau=tau)
    expected = np.concatenate([data[..., None, j + tau * np.arange(d)] for j in
                               range(data.shape[-1] - (d - 1) * tau)],
                              axis=data.ndim - 1)
    assert_almost_equal(emb_data, expected)


def test_filt():
    filt_low_pass = _filt(sfreq, data, [None, 50.])
    filt_bandpass = _filt(sfreq, data, [1., 70.])
    assert_equal(filt_low_pass.shape, data.shape)
    assert_equal(filt_bandpass.shape, data.shape)


if __name__ == '__main__':

    test_power_spectrum()
    test_psd()
    test_idxiter()
    test_embed()
    test_filt()
