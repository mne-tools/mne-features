import warnings
import numpy as np
from mne_features.univariate import get_univariate_funcs
from mne_features.bivariate import get_bivariate_funcs
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion


def _check_func_names(selected, feature_funcs_names):
    """ Checks if the names of selected feature functions match the available
    feature functions.

    Parameters
    ----------
    selected : list of str
        Names of the selected feature functions.

    feature_funcs_names : dict-keys or list
        Names of available feature functions.

    Returns
    -------
    valid_func_names : list of str
    """
    valid_func_names = list()
    for f in selected:
        if f in feature_funcs_names:
            valid_func_names.append(f)
        else:
            warnings.warn('The name ``%s`` is not a valid feature function. '
                          'This name was ignored.' % f)
    if len(valid_func_names) == 0:
        raise ValueError('No valid feature function names given.')
    else:
        return valid_func_names


def extract_features(X, sfreq, selected_funcs):
    """ Extraction of temporal or spectral features from epoched EEG signals.

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_times)
        Array of epoched EEG data.

    sfreq : float
        Sampling rate of the data.

    selected_funcs : list of str
        The elements of `selected_features` are the names of the feature
        functions which will be used to extract features from the data.
        The available feature functions are :
            - 'kurtosis'
            - 'max_cross_corr' : maximum cross-correlation
            - 'mean'
            - 'ptp_amplitude' : peak-to-peak amplitude of the signal
            - 'skewness'
            - 'variance'

    Returns
    -------
    ndarray, shape (n_epochs, n_features)
    """
    if sfreq <= 0:
        raise ValueError('Sampling rate `sfreq` must be positive.')
    univariate_funcs = get_univariate_funcs()
    bivariate_funcs = get_bivariate_funcs(sfreq)
    feature_funcs = univariate_funcs.copy()
    feature_funcs.update(bivariate_funcs)
    sel_funcs = _check_func_names(selected_funcs, feature_funcs.keys())

    # Feature extraction
    n_epochs = X.shape[0]
    Xnew = list()
    _tr = [(n, FunctionTransformer(func=feature_funcs[n])) for n in sel_funcs]
    extractor = FeatureUnion(transformer_list=_tr)
    for j in range(n_epochs):
        features_epoch = extractor.fit_transform(X[j, :, :])
        Xnew.append(features_epoch)
    Xnew = np.vstack(Xnew)
    return Xnew
