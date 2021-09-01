# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

"""Bivariate feature functions."""

from math import sqrt

import numpy as np
from scipy import signal
from scipy.spatial.distance import pdist, squareform
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import scale

from .mock_numba import nb
from .utils import (_idxiter, power_spectrum, _embed, _get_feature_funcs,
                    _get_feature_func_names, _psd_params_checker)


def get_bivariate_funcs(sfreq):
    """Mapping between aliases and bivariate feature functions.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    Returns
    -------
    bivariate_funcs : dict
    """
    return _get_feature_funcs(sfreq, __name__)


def get_bivariate_func_names():
    """List of names of bivariate feature functions.

    Returns
    -------
    bivariate_func_names : list
    """
    return _get_feature_func_names(__name__)


@nb.jit([nb.float64[:](nb.float64, nb.float64[:, :], nb.optional(nb.boolean)),
         nb.float32[:](nb.float32, nb.float32[:, :], nb.optional(nb.boolean))],
        nopython=True)
def _max_cross_corr(sfreq, data, include_diag=False):
    """Utility function for :func:`compute_max_cross_correlation`.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    data : ndarray, shape (n_channels, n_times)
        The signals.

    include_diag : bool (default: False)
        If False, features corresponding to pairs of identical electrodes
        are not computed. In other words, features are not computed from pairs
        of electrodes of the form ``(ch[i], ch[i])``.

    Returns
    -------
    output : ndarray, shape (n_output,)
        With ``n_output = n_channels * (n_channels + 1) / 2`` if
        ``include_diag`` is True and
        ``n_output = n_channels * (n_channels - 1) / 2`` if
        ``include_diag`` is False.
    """
    n_channels, n_times = data.shape
    n_tau = int(0.5 * sfreq)
    taus = np.arange(-n_tau, n_tau)
    if include_diag:
        n_coefs = n_channels * (n_channels + 1) // 2
    else:
        n_coefs = n_channels * (n_channels - 1) // 2
    max_cc = np.empty((n_coefs,), dtype=data.dtype)
    for s, k, l in _idxiter(n_channels, include_diag=include_diag):
        max_cc_ij = np.empty((2 * n_tau,))
        for tau in taus:
            if tau < 0:
                _tau = -tau
            else:
                _tau = tau
            x_m = 0
            y_m = 0
            for j in range(n_times):
                x_m += data[k, j]
                y_m += data[l, j]
            x_m /= n_times
            y_m /= n_times
            x_v = 0
            y_v = 0
            for j in range(n_times):
                x_v += (data[k, j] - x_m) * (data[k, j] - x_m)
                y_v += (data[l, j] - y_m) * (data[l, j] - y_m)
            x_v /= (n_times - 1)
            y_v /= (n_times - 1)
            x_v = sqrt(x_v)
            y_v = sqrt(y_v)
            cc = 0
            for j in range(0, n_times - _tau):
                cc += ((data[k, j + _tau] - x_m) / x_v) * ((data[l, j] -
                                                            y_m) / y_v)
            cc /= (n_times - _tau)
            max_cc_ij[tau + n_tau] = abs(cc)
        max_cc[s] = np.max(max_cc_ij)
    return max_cc


def compute_max_cross_corr(sfreq, data, include_diag=False):
    """Maximum linear cross-correlation.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    data : ndarray, shape (n_channels, n_times)
        The signals.

    include_diag : bool (default: False)
        If False, features corresponding to pairs of identical electrodes
        are not computed. In other words, features are not computed from pairs
        of electrodes of the form ``(ch[i], ch[i])``.

    Returns
    -------
    output : ndarray, shape (n_output,)
        With ``n_output = n_channels * (n_channels + 1) / 2`` if
        ``include_diag`` is True and
        ``n_output = n_channels * (n_channels - 1) / 2`` if
        ``include_diag`` is False.

    Notes
    -----
    Alias of the feature function: **max_cross_corr**. See [1]_ and [2]_.

    References
    ----------
    .. [1] Mormann, F. et al. (2006). Seizure prediction: the long and
           winding road. Brain, 130(2), 314-333.

    .. [2] Mirowski, P. W. et al. (2008). Comparing SVM and convolutional
           networks for epileptic seizure prediction from intracranial EEG.
           Machine Learning for Signal Processing, 2008. IEEE Workshop on
           (pp. 244-249). IEEE.
    """
    return _max_cross_corr(sfreq, data, include_diag=include_diag)


def compute_phase_lock_val(data, include_diag=False):
    """Phase Locking Value (PLV).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    include_diag : bool (default: False)
        If False, features corresponding to pairs of identical electrodes
        are not computed. In other words, features are not computed from pairs
        of electrodes of the form ``(ch[i], ch[i])``.

    Returns
    -------
    output : ndarray, shape (n_output,)
        With ``n_output = n_channels * (n_channels + 1) / 2`` if
        ``include_diag`` is True and
        ``n_output = n_channels * (n_channels - 1) / 2`` if
        ``include_diag`` is False.

    Notes
    -----
    Alias of the feature function: **phase_lock_val**. See [1]_.

    References
    ----------
    .. [1] http://www.gatsby.ucl.ac.uk/~vincenta/kaggle/report.pdf
    """
    n_channels, n_times = data.shape
    if include_diag:
        n_coefs = n_channels * (n_channels + 1) // 2
    else:
        n_coefs = n_channels * (n_channels - 1) // 2
    plv = np.empty((n_coefs,))
    for s, i, j in _idxiter(n_channels, include_diag=include_diag):
        if i == j:
            plv[j] = 1
        else:
            xa = signal.hilbert(data[i, :])
            ya = signal.hilbert(data[j, :])
            phi_x = np.angle(xa)
            phi_y = np.angle(ya)
            plv[s] = np.absolute(np.mean(np.exp(1j * (phi_x - phi_y))))
    return plv


def compute_nonlin_interdep(data, tau=2, emb=10, nn=5, include_diag=False):
    """Measure of nonlinear interdependence.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        The signals.

    tau : int (default: 2)
        Delay in time samples.

    emb : int (default: 10)
        Embedding dimension.

    nn : int (default: 5)
        Number of Nearest Neighbors.

    include_diag : bool (default: False)
        If False, features corresponding to pairs of identical electrodes
        are not computed. In other words, features are not computed from pairs
        of electrodes of the form ``(ch[i], ch[i])``.

    Returns
    -------
    output : ndarray, shape (n_output,)
        With ``n_output = n_channels * (n_channels + 1) / 2`` if
        ``include_diag`` is True and
        ``n_output = n_channels * (n_channels - 1) / 2`` if
        ``include_diag`` is False.

    Notes
    -----
    Alias of the feature function: **nonlin_interdep**. See [1]_ and [2]_.

    References
    ----------
    .. [1] Mormann, F. et al. (2006). Seizure prediction: the long and
           winding road. Brain, 130(2), 314-333.

    .. [2] Mirowski, P. W. et al. (2008). Comparing SVM and convolutional
           networks for epileptic seizure prediction from intracranial EEG.
           Machine Learning for Signal Processing, 2008. IEEE Workshop on
           (pp. 244-249). IEEE.
    """
    n_channels, n_times = data.shape
    if include_diag:
        n_coefs = n_channels * (n_channels + 1) // 2
    else:
        n_coefs = n_channels * (n_channels - 1) // 2
    nlinterdep = np.empty((n_coefs,))
    for s, i, j in _idxiter(n_channels, include_diag=include_diag):
        emb_x = _embed(data[i, None], d=emb, tau=tau)[0, :, :]
        emb_y = _embed(data[j, None], d=emb, tau=tau)[0, :, :]
        knn = NearestNeighbors(n_neighbors=nn, algorithm='kd_tree')
        idx_x = clone(knn).fit(emb_x).kneighbors(emb_x, return_distance=False)
        idx_y = clone(knn).fit(emb_y).kneighbors(emb_y, return_distance=False)
        gx = squareform(pdist(emb_x, metric='sqeuclidean'))
        gy = squareform(pdist(emb_y, metric='sqeuclidean'))
        nr = gx.shape[0]
        rx = np.mean(np.vstack([gx[j, idx_x[j, :]] for j in range(nr)]))
        rxy = np.mean(np.vstack([gx[j, idx_y[j, :]] for j in range(nr)]))
        ry = np.mean(np.vstack([gy[j, idx_y[j, :]] for j in range(nr)]))
        ryx = np.mean(np.vstack([gy[j, idx_x[j, :]] for j in range(nr)]))
        sxy = np.mean(np.divide(rx, rxy))
        syx = np.mean(np.divide(ry, ryx))
        nlinterdep[s] = sxy + syx
    return nlinterdep


def compute_time_corr(data, with_eigenvalues=False, include_diag=True):
    """Correlation Coefficients (computed in the time domain).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        The signals.

    with_eigenvalues : bool (default: False)
        If True, the function also returns the eigenvalues of the correlation
        matrix.

    include_diag : bool (default: True)
        If False, features corresponding to pairs of identical electrodes
        are not computed. In other words, features are not computed from pairs
        of electrodes of the form ``(ch[i], ch[i])``.

    Returns
    -------
    output : ndarray, shape (n_output,)
        With ``n_output = n_coefs + n_channels`` if ``with_eigenvalues`` is
        True and ``n_output = n_coefs`` if ``with_eigenvalues`` is False. If
        ``include_diag`` is True, then
        ``n_coefs = n_channels * (n_channels + 1) // 2`` and
        ``n_coefs = n_channels * (n_channels - 1) // 2`` otherwise.

    Notes
    -----
    Alias of the feature function: **time_corr**. See [1]_.

    References
    ----------
    .. [1] https://kaggle2.blob.core.windows.net/forum-message-attachments/
           134445/4803/seizure-detection.pdf
    """
    n_channels = data.shape[0]
    _scaled = scale(data, axis=0)
    corr = np.corrcoef(_scaled)
    coefs = corr[np.triu_indices(n_channels, 1 - int(include_diag))]
    if with_eigenvalues:
        w, _ = np.linalg.eig(corr)
        w = np.abs(w)
        w = np.sort(w)
        return np.r_[coefs, w]
    else:
        return coefs


def _compute_eig_feat_names(data, with_eigenvalues, include_diag, **kwargs):
    """Utility function to create feature names compatible with the output of
    :func:`mne_features.bivariate.compute_time_corr` and
    :func:`mne_features.bivariate.compute_spect_corr`."""
    feat_names = [f'ch{i}-ch{j}' for _, i, j, in _idxiter(
                  data.shape[0], include_diag=include_diag)]
    if with_eigenvalues:
        feat_names.extend([f'eig{i}' for i in range(data.shape[0])])
    return feat_names


compute_time_corr.get_feature_names = _compute_eig_feat_names


def compute_spect_corr(sfreq, data, with_eigenvalues=True,
                       include_diag=False, psd_method='welch',
                       psd_params=None):
    """Correlation Coefficients (computed from the power spectrum).

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    data : ndarray, shape (n_channels, n_times)
        The signals.

    with_eigenvalues : bool (default: True)
        If True, the function also returns the eigenvalues of the correlation
        matrix.

    include_diag : bool (default: False)
        If False, features corresponding to pairs of identical electrodes
        are not computed. In other words, features are not computed from pairs
        of electrodes of the form ``(ch[i], ch[i])``.

    psd_method : str (default: 'welch')
        Method used for the estimation of the Power Spectral Density (PSD).
        Valid methods are: ``'welch'``, ``'multitaper'`` or ``'fft'``.

    psd_params : dict or None (default: None)
        If not None, dict with optional parameters (`welch_n_fft`,
        `welch_n_per_seg`, `welch_n_overlap`) to be passed to
        :func:`mne_features.utils.power_spectrum`. If None, default parameters
        are used (see doc for :func:`mne_features.utils.power_spectrum`).

    Returns
    -------
    output : ndarray, shape (n_output,)
        Where ``n_output = n_coefs + n_channels`` if ``with_eigenvalues`` is
        True and ``n_output = n_coefs`` if ``with_eigenvalues`` is False. If
        ``include_diag`` is True, then
        ``n_coefs = n_channels * (n_channels + 1) // 2`` and
        ``n_coefs = n_channels * (n_channels - 1) // 2`` otherwise.

    Notes
    -----
    Alias of the feature function: **spect_corr**. See [1]_.

    References
    ----------
    .. [1] https://kaggle2.blob.core.windows.net/forum-message-attachments/
           134445/4803/seizure-detection.pdf
    """
    n_channels = data.shape[0]
    _psd_params = _psd_params_checker(psd_params)
    ps, _ = power_spectrum(sfreq, data, psd_method=psd_method, **_psd_params)
    _scaled = scale(ps, axis=0)
    corr = np.corrcoef(_scaled)
    coefs = corr[np.triu_indices(n_channels, 1 - int(include_diag))]
    if with_eigenvalues:
        w, _ = np.linalg.eig(corr)
        w = np.abs(w)
        w = np.sort(w)
        return np.r_[coefs, w]
    else:
        return coefs


compute_spect_corr.get_feature_names = _compute_eig_feat_names
