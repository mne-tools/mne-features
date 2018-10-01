# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_raises
from scipy import signal

from mne_features.utils import (_idxiter, power_spectrum, _embed, _filt,
                                _psd_params_checker)

rng = np.random.RandomState(42)
sfreq = 172.
data = rng.standard_normal((20, int(sfreq)))


def test_psd():
    n_channels, n_times = data.shape
    _data = data[None, ...]
    # Only test output shape when `method='welch'` or `method='multitaper'`
    # since it is actually just a wrapper for MNE functions:
    psd_welch, _ = power_spectrum(sfreq, _data, psd_method='welch')
    psd_multitaper, _ = power_spectrum(sfreq, _data, psd_method='multitaper')
    psd_fft, freqs_fft = power_spectrum(sfreq, _data, psd_method='fft')
    assert_equal(psd_welch.shape, (1, n_channels, n_times // 2 + 1))
    assert_equal(psd_multitaper.shape, (1, n_channels, n_times // 2 + 1))
    assert_equal(psd_fft.shape, (1, n_channels, n_times // 2 + 1))

    # Compare result obtained with `method='fft'` to the Scipy's result
    # (implementation of Welch's method with rectangular window):
    expected_freqs, expected_psd = signal.welch(data, sfreq,
                                                window=signal.get_window(
                                                    'boxcar', data.shape[-1]),
                                                return_onesided=True,
                                                scaling='density')
    assert_almost_equal(expected_freqs, freqs_fft)
    assert_almost_equal(expected_psd, psd_fft[0, ...])


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


def test_psd_params_checker():
    valid_params = {'welch_n_fft': 2048, 'welch_n_per_seg': 1024}
    assert_equal(valid_params, _psd_params_checker(valid_params))
    assert_equal(dict(), _psd_params_checker(None))
    with assert_raises(ValueError):
        invalid_params1 = {'n_fft': 1024, 'psd_method': 'fft'}
        _psd_params_checker(invalid_params1)
    with assert_raises(ValueError):
        invalid_params2 = [1024, 1024]
        _psd_params_checker(invalid_params2)


if __name__ == '__main__':

    test_psd()
    test_idxiter()
    test_embed()
    test_filt()
    test_psd_params_checker()
