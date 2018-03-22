# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


from inspect import getargs

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer

from .bivariate import get_bivariate_funcs
from .univariate import get_univariate_funcs


class FeatureFunctionTransformer(FunctionTransformer):
    """ Construct a transformer from a given feature function.

    Similarly to FunctionTransformer, FeatureFunctionTranformer applies a
    feature function to a given array X.

    Parameters
    ----------
    func : callable or None (default: None)
        Feature function to be used in the transformer.
        If None, the identity function is used.

    validate : bool (default: True)
        If True, the array X will be checked before calling the function.
        If possible, a 2d Numpy array is returned. Otherwise, an exception
        will be raised. If False, the array X is not checked.

    kw_args : dict or None (default: None)
        If not None, dictionary of additional keyword arguments to pass to the
        feature function.
    """
    def __init__(self, func=None, validate=True, kw_args=None):
        super(FeatureFunctionTransformer, self).__init__(func=func,
                                                         validate=validate,
                                                         kw_args=kw_args)

    def transform(self, X, y='deprecated'):
        """ Transform the array X with the given feature function.

        Parameters
        ----------
        X : ndarray, shape (n_channels, n_times)

        y : (ignored)

        Returns
        -------
        X_out : ndarray, shape (n_output_func,)
            Usually, `n_output_func` will be equal to `n_channels` for most
            univariate feature functions and to
            `(n_channels * (n_channels + 1)) // 2` for most bivariate feature
            functions. See the doc of `func` for more details.
        """
        X_out = super(FeatureFunctionTransformer, self).transform(X, y)
        self.output_shape_ = X_out.shape[0]
        return X_out

    def get_feature_names(self):
        """ Mapping of the feature indices to feature names. """
        if not hasattr(self, 'output_shape_'):
            raise ValueError('Call `transform` or `fit_transform` first.')
        else:
            return np.arange(self.output_shape_).astype(str)

    def get_params(self, deep=True):
        """ Get the parameters (if any) of the given feature function.

        Parameters
        ----------
        deep : bool (default: True)
            If True, the method will get the parameters of the transformer and
            subobjects. (See `sklearn.preprocessing.FunctionTransformer`).
        """
        _params = super(FeatureFunctionTransformer, self).get_params(deep=deep)
        if hasattr(_params['func'], 'func'):
            # If `_params['func'] is of type `functools.partial`
            _to_inspect = _params['func'].func
        elif hasattr(_params['func'], 'py_func'):
            # If `_params['func'] is a jitted Python function
            _to_inspect = _params['func'].py_func
        else:
            # If `_params['func'] is an actual Python function
            _to_inspect = _params['func']
        args, _, _ = getargs(_to_inspect.func_code)
        defaults = _to_inspect.func_defaults
        if defaults is None:
            return dict()
        else:
            n_defaults = len(defaults)
            func_params = {key: value for key, value in
                           zip(args[-n_defaults:], defaults)}
        return func_params

    def set_params(self, **params):
        """ Set the parameters of the given feature function. """
        self.kw_args = params
        return self


def _format_as_dataframe(X, feature_names):
    """ Utility function to format extracted features (X) as a Pandas
    DataFrame using names and indexes from `feature_names`. The index of the
    columns is a MultiIndex with two levels. At level 0, the alias of the
    feature function is given. At level 1, an enumeration of the features is
    given.

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_features)
        Extracted features. `X` should be the output of `extract_features`.

    feature_names : list of str

    Returns
    -------
    output : Pandas DataFrame
    """
    n_features = X.shape[1]
    if len(feature_names) != n_features:
        raise ValueError('The length of `feature_names` should be equal to '
                         '`X.shape[1]` (`n_features`).')
    else:
        _names = [n.split('__')[0] for n in feature_names]
        _idx = [n.split('__')[1] for n in feature_names]
        columns = pd.MultiIndex.from_arrays([_names, _idx])
        return pd.DataFrame(data=X, columns=columns)


def _apply_extractor(extractor, X):
    """ Utility function to apply features extractor to ndarray X.

    Parameters
    ----------
    extractor : Instance of sklearn.pipeline.FeatureUnion or sklearn.pipeline

    X : ndarray, shape (n_channels, n_times)

    Returns
    -------
    ndarray, shape (n_features,)
    """
    return extractor.fit_transform(X)


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
            ValueError('The given alias (%s) is not valid. The valid aliases '
                       'for feature functions are: %s.' %
                       (f, feature_funcs_names))
    if not valid_func_names:
        raise ValueError('No valid feature function names given.')
    else:
        return valid_func_names


def extract_features(X, sfreq, freq_bands, selected_funcs, funcs_params=None,
                     n_jobs=1, return_as_df=False):
    """ Extraction of temporal or spectral features from epoched EEG signals.

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_times)
        Array of epoched EEG data.

    sfreq : float
        Sampling rate of the data.

    freq_bands : ndarray, shape (n_freqs,)
        Array defining the frequency bands. The j-th frequency band is defined
        as: [freq_bands[j], freq_bands[j + 1]] (0 <= j <= n_freqs - 1).

    selected_funcs : list of str
        The elements of `selected_features` are aliases for the feature
        functions which will be used to extract features from the data.
        (See `mne_features` documentation for a complete list of available
        feature functions).

    funcs_params : dict or None (default: None)
        If not None, dict of optional parameters to be passed to the feature
        functions. Each key of the `funcs_params` dict should be of the form :
        [alias_feature_function]__[optional_param] (for example:
        'higuchi_fd__kmax`).

    n_jobs : int (default: 1)
        Number of CPU cores used when parallelizing the feature extraction.
        If given a value of -1, all cores are used.

    return_as_df : bool (default: False)
        If True, the extracted features will be returned as a Pandas DataFrame.
        The column index is a MultiIndex (see `pd.MultiIndex`) which contains
        the alias of each feature function which was used. If False, the
        features are returned as a 2d Numpy array.

    Returns
    -------
    array-like, shape (n_epochs, n_features)
    """
    if sfreq <= 0:
        raise ValueError('Sampling rate `sfreq` must be positive.')
    univariate_funcs = get_univariate_funcs(sfreq, freq_bands)
    bivariate_funcs = get_bivariate_funcs(sfreq)
    feature_funcs = univariate_funcs.copy()
    feature_funcs.update(bivariate_funcs)
    sel_funcs = _check_func_names(selected_funcs, feature_funcs.keys())

    # Feature extraction
    n_epochs = X.shape[0]
    _tr = [(n, FeatureFunctionTransformer(func=feature_funcs[n]))
           for n in sel_funcs]
    extractor = FeatureUnion(transformer_list=_tr)
    if funcs_params is not None:
        extractor.set_params(**funcs_params)
    res = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_apply_extractor)(
        extractor, X[j, :, :]) for j in range(n_epochs))
    Xnew = np.vstack(res)
    if return_as_df:
        return _format_as_dataframe(Xnew, extractor.get_feature_names())
    else:
        return Xnew
