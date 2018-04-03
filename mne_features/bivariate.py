# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


from functools import partial
from math import sqrt

import numpy as np
from scipy import signal
from scipy.spatial.distance import pdist, squareform
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import scale

from .mock_numba import nb
from .utils import triu_idx, power_spectrum, embed


def get_bivariate_funcs(sfreq):
    """ Returns a dictionary of bivariate feature functions. For each feature
    function, the corresponding key in the dictionary is an alias for the
    function.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    Returns
    -------
    bivariate_funcs : dict of feature functions
    """
    bivariate_funcs = dict()
    bivariate_funcs['max_cross_corr'] = partial(compute_max_cross_correlation,
                                                sfreq)
    bivariate_funcs['plv'] = compute_phase_locking_value
    bivariate_funcs['nonlin_interdep'] = compute_nonlinear_interdep
    bivariate_funcs['time_corr'] = compute_time_corr_coefs
    bivariate_funcs['spect_corr'] = partial(compute_spect_corr_coefs, sfreq)
    return bivariate_funcs


@nb.jit([nb.float64[:](nb.float64, nb.float64[:, :]),
         nb.float32[:](nb.float32, nb.float32[:, :])], nopython=True)
def compute_max_cross_correlation(s_freq, data):
    """ Maximum linear cross-correlation [1, 2].

    Parameters
    ----------
    s_freq : float
        Sampling rate of the data.

    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels * (n_channels + 1) / 2,)

    References
    ----------
    .. [1] Mormann, F. et al. (2006). Seizure prediction: the long and winding
           road. Brain, 130(2), 314-333.

    .. [2] Mirowski, P. W. et al. (2008). Comparing SVM and convolutional
           networks for epileptic seizure prediction from intracranial EEG.
           Machine Learning for Signal Processing, 2008.
           IEEE Workshop on (pp. 244-249). IEEE.
    """

    n_channels, n_times = data.shape
    n_tau = int(0.5 * s_freq)
    taus = np.arange(-n_tau, n_tau)
    n_coefs = n_channels * (n_channels + 1) // 2
    max_cc = np.empty((n_coefs,), dtype=data.dtype)
    for s, k, l in triu_idx(n_channels):
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


def compute_phase_locking_value(data):
    """ Phase Locking Value (PLV) [1].

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels * (n_channels + 1) / 2,)

    References
    ----------
    .. [1] http://www.gatsby.ucl.ac.uk/~vincenta/kaggle/report.pdf
    """
    n_channels, n_times = data.shape
    n_coefs = n_channels * (n_channels + 1) // 2
    plv = np.empty((n_coefs,))
    for s, i, j in triu_idx(n_channels):
        if i == j:
            plv[j] = 1
        else:
            xa = signal.hilbert(data[i, :])
            ya = signal.hilbert(data[j, :])
            phi_x = np.angle(xa)
            phi_y = np.angle(ya)
            plv[s] = np.absolute(np.mean(np.exp(1j * (phi_x - phi_y))))
    return plv


def compute_nonlinear_interdep(data, tau=2, emb=10, nn=5):
    """ Measure of nonlinear interdependence [1, 2].

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    tau : int (default: 2)
        Delay.

    emb : int (default: 10)
        Embedding dimension.

    nn : int (default: 5)
        Number of Nearest Neighbours.

    Returns
    -------
    ndarray, shape (n_channels * (n_channels + 1) / 2,)

    References
    ----------
    .. [1] Mormann, F. et al. (2006). Seizure prediction: the long and winding
           road. Brain, 130(2), 314-333.

    .. [2] Mirowski, P. W. et al. (2008). Comparing SVM and convolutional
           networks for epileptic seizure prediction from intracranial EEG.
           In Machine Learning for Signal Processing. IEEE. pp. 244-249.
    """
    n_channels, n_times = data.shape
    n_coefs = n_channels * (n_channels + 1) // 2
    nlinterdep = np.empty((n_coefs,))
    for s, i, j in triu_idx(n_channels):
        emb_x = embed(data[i, :], d=emb, tau=tau)
        emb_y = embed(data[j, :], d=emb, tau=tau)
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


def compute_time_corr_coefs(data, with_eigenvalues=True):
    """ Correlation Coefficients (computed in the time domain) [1].

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    with_eigenvalues : bool (default: True)
        If True, the function also returns the eigenvalues of the correlation
        matrix.

    Returns
    -------
    output : ndarray, shape (n_out,)
        If `with_eigenvalues` is True, n_out = n_coefs + n_channels (with:
        n_coefs = n_channels * (n_channels + 1) // 2). Otherwise,
        n_out = n_coefs.

    References
    ----------
    .. [1] https://kaggle2.blob.core.windows.net/forum-message-attachments/
           134445/4803/seizure-detection.pdf
    """
    n_channels = data.shape[0]
    _scaled = scale(data, axis=0)
    corr = np.corrcoef(_scaled)
    coefs = corr[np.triu_indices(n_channels)]
    if with_eigenvalues:
        w, _ = np.linalg.eig(corr)
        w = np.abs(w)
        w = np.sort(w)
        return np.r_[coefs, w]
    else:
        return coefs


def compute_spect_corr_coefs(sfreq, data, db=False, with_eigenvalues=True):
    """ Correlation Coefficients (computed from the power spectrum) [1].

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    data : ndarray, shape (n_channels, n_times)

    db : bool (default: True)
        If True, the power spectrum returned by the function
        `compute_power_spectrum` is returned in dB/Hz.

    with_eigenvalues : bool (default: True)
        If True, the function also returns the eigenvalues of the correlation
        matrix.

    Returns
    -------
    output : ndarray, shape (n_out,)
        If `with_eigenvalues` is True, n_out = n_coefs + n_channels. Otherwise,
        n_out = n_coefs. With, n_coefs = n_channels * (n_channels + 1) // 2.

    References
    ----------
    .. [1] https://kaggle2.blob.core.windows.net/forum-message-attachments/
           134445/4803/seizure-detection.pdf
    """
    n_channels = data.shape[0]
    ps, _ = power_spectrum(sfreq, data, return_db=db)
    _scaled = scale(ps, axis=0)
    corr = np.corrcoef(_scaled)
    coefs = corr[np.triu_indices(n_channels)]
    if with_eigenvalues:
        w, _ = np.linalg.eig(corr)
        w = np.abs(w)
        w = np.sort(w)
        return np.r_[coefs, w]
    else:
        return coefs
