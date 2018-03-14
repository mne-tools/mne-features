# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


from functools import partial
from math import sqrt

import numpy as np
from scipy import signal
from sklearn.preprocessing import scale

from .mock_numba import nb
from .univariate import power_spectrum


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
    bivariate_funcs['spect_corr'] = compute_spect_corr_coefs
    return bivariate_funcs


@nb.jit()
def _triu_idx(n):
    """ Utility function to generate an enumeration of the pairs of indices
    (i,j) corresponding to the upper triangular part of a (n, n) array.

    Parameters
    ----------
    n : int

    Returns
    -------
    generator
    """
    pos = -1
    for i in range(n):
        for j in range(i, n):
            pos += 1
            yield pos, i, j


def compute_max_cross_correlation(s_freq, data):
    """ Maximum linear cross-correlation [1, 2].

    Parameters
    ----------
    s_freq : float
        Sampling rate of the data.

    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    ndarray, shape (n_channels * (n_channels + 1) / 2,)

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
    taus = np.arange(-n_tau, n_tau, dtype=np.int64)
    n_coefs = n_channels * (n_channels + 1) // 2
    max_cc = np.empty((n_coefs,))
    for s, k, l in _triu_idx(n_channels):
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
        max_cc[s] = max(max_cc_ij)
    return max_cc


def compute_phase_locking_value(data):
    """ Phase Locking Value (PLV) [1].

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    ndarray, shape (n_channels * (n_channels + 1) / 2,)

    References
    ----------
    .. [1] http://www.gatsby.ucl.ac.uk/~vincenta/kaggle/report.pdf
    """
    n_channels, n_times = data.shape
    n_coefs = n_channels * (n_channels + 1) / 2
    plv = np.empty((n_coefs,))
    for s, i, j in _triu_idx(n_channels):
        if i == j:
            plv[j] = 1
        else:
            xa = signal.hilbert(data[i, :])
            ya = signal.hilbert(data[j, :])
            phi_x = np.angle(xa)
            phi_y = np.angle(ya)
            plv[s] = np.absolute(np.mean(np.exp(1j * (phi_x - phi_y))))
    return plv


@nb.jit([nb.float64[:](nb.float64[:, :], nb.optional(nb.int64),
                       nb.optional(nb.int64), nb.optional(nb.int64),
                       nb.optional(nb.int64)),
         nb.float32[:](nb.float32[:, :], nb.optional(nb.int32),
                       nb.optional(nb.int32), nb.optional(nb.int32),
                       nb.optional(nb.int32))])
def compute_nonlinear_interdep(data, tau=2, emb=10, theiler=50, nn=10):
    """ Measure of nonlinear interdependence [1, 2, 3].

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    tau : int (default: 2)
        Delay (number of samples)

    emb : int (default: 10)
        Embedding dimension.

    theiler : int (default: 50)
        Theiler correction (number of samples)

    nn : int (default: 10)
        Number of Nearest Neighbours.

    Returns
    -------
    ndarray, shape (n_channels * (n_channels + 1) / 2,)

    References
    ----------
    .. [1] Quiroga, R. Q. et al. (2002). Performance of different
           synchronization measures in real data: a case study on
           electroencephalographic signals. Physical Review E, 65(4), 041903.

    .. [2] https://vis.caltech.edu/~rodri/software.htm

    .. [3] Mirowski, P. W. et al. (2008). Comparing SVM and convolutional
           networks for epileptic seizure prediction from intracranial EEG.
           In Machine Learning for Signal Processing. IEEE. pp. 244-249.
    """
    n_channels, n_times = data.shape
    n_coefs = n_channels * (n_channels + 1) / 2
    nlinterdep = np.empty((n_coefs,))
    for s, u, v in _triu_idx(n_channels):
        aux_x = np.zeros((nn + 1,))
        aux_y = np.zeros((nn + 1,))
        dist_x = np.zeros((n_times,))
        dist_y = np.zeros((n_times,))
        idx_x = np.zeros((nn + 1,))
        idx_y = np.zeros((nn + 1,))
        sxy = 0
        syx = 0
        for i in range(n_times - (emb - 1) * tau):
            # Initialize aux_x, aux_y, idx_x, idx_y to large values
            for k in range(nn):
                aux_x[k] = 100000000
                aux_y[k] = 100000000
                idx_x[k] = 100000000
                idx_y[k] = 100000000
            aux_x[nn] = 0
            aux_y[nn] = 0
            idx_x[nn] = 10000000
            idx_y[nn] = 10000000
            rrx = 0
            rry = 0
            for j in range(n_times - (emb - 1) * tau):
                dist_x[j] = 0
                dist_y[j] = 0
                for k in range(emb):
                    dist_x[j] += ((data[u, i + k * tau] -
                                   data[u, j + k * tau]) *
                                  (data[u, i + k * tau] -
                                   data[u, j + k * tau]))
                    dist_y[j] += ((data[v, i + k * tau] -
                                   data[v, j + k * tau]) *
                                  (data[v, i + k * tau] -
                                   data[v, j + k * tau]))
                if abs(i - j) > theiler:
                    if dist_x[j] < aux_x[0]:
                        for k in range(nn + 1):
                            if dist_x[j] < aux_x[k]:
                                aux_x[k] = aux_x[k + 1]
                                idx_x[k] = idx_x[k + 1]
                            else:
                                aux_x[k - 1] = dist_x[j]
                                idx_x[k - 1] = j
                                break
                    if dist_y[j] < aux_y[0]:
                        for k in range(nn + 1):
                            if dist_y[j] < aux_y[k]:
                                aux_y[k] = aux_y[k + 1]
                                idx_y[k] = idx_y[k + 1]
                            else:
                                aux_y[k - 1] = dist_y[j]
                                idx_y[k - 1] = j
                                break
                rrx += dist_x[j]
                rry += dist_y[j]
            rxx = 0
            ryy = 0
            rxy = 0
            ryx = 0
            for k in range(nn):
                rxx += aux_x[k]
                ryy += aux_y[k]
                rxy += dist_x[int(idx_y[k])]
                ryx += dist_y[int(idx_x[k])]
            rxx /= nn
            ryy /= nn
            rxy /= nn
            ryx /= nn
            sxy += rxx / rxy
            syx += ryy / ryx
        sxy /= (n_times - (emb - 1) * tau)
        syx /= (n_times - (emb - 1) * tau)
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
    ndarray, shape (n_out,)
        If `with_eigenvalues` is True, n_out = n_coefs + n_channels (with:
        n_coefs = n_channels * (n_channels + 1) / 2). Otherwise,
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


def compute_spect_corr_coefs(sfreq, data, db=True, with_eigenvalues=True):
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
    ndarray, shape (n_out,)
        If `with_eigenvalues` is True, n_out = n_coefs + n_channels. Otherwise,
        n_out = n_coefs. With, n_coefs = n_channels * (n_channels + 1) / 2.

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
