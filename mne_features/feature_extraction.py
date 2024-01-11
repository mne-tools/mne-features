"""Feature extraction functions."""

# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

import re
from inspect import getargs

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
try:
    from sklearn.externals import joblib
except (ImportError, ModuleNotFoundError):
    import joblib
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer

from .bivariate import get_bivariate_funcs, get_bivariate_func_names
from .univariate import get_univariate_funcs, get_univariate_func_names
from .utils import _get_func_name, _get_python_func, _idxiter


class FeatureFunctionTransformer(FunctionTransformer):
    """Construct a transformer from a given feature function.

    Similarly to :class:`~sklearn.preprocessing.FunctionTransformer`,
    :class:`FeatureFunctionTranformer` applies a feature function to a given
    array X.

    Parameters
    ----------
    func : callable or None (default: None)
        Feature function to be used in the transformer.
        If None, the identity function is used.

    validate : bool (default: True)
        If True, the array X will be checked before calling the function.
        If possible, a 2d Numpy array is returned. Otherwise, an exception
        will be raised. If False, the array X is not checked.

    params : dict or None (default: None)
        If not None, dictionary of additional keyword arguments to pass to the
        feature function.
    """

    def __init__(self, func=None, validate=True, params=None):
        """Instantiate a FeatureFunctionTransformer object."""
        self.params = params
        super(FeatureFunctionTransformer, self).__init__(func=func,
                                                         validate=validate,
                                                         kw_args=params)

    def transform(self, X):
        """Apply the given feature function to the array X.

        Parameters
        ----------
        X : ndarray, shape (n_channels, n_times)

        Returns
        -------
        X_out : ndarray, shape (n_output_func,)
            Usually, ``n_output_func`` will be equal to ``n_channels`` for most
            univariate feature functions and to
            ``(n_channels * (n_channels + 1)) // 2`` for most bivariate feature
            functions. See the doc of the given feature function for more
            details.
        """
        X_out = super(FeatureFunctionTransformer, self).transform(X)
        self.output_shape_ = X_out.shape[0]
        if not hasattr(self, 'feature_names_'):
            func_name = _get_func_name(self.func).replace('compute_', '')
            if (func_name in get_univariate_func_names() and
                    self.output_shape_ == X.shape[0]):
                self.feature_names_ = ['ch%s' % ch
                                       for ch in range(self.output_shape_)]
            elif func_name in get_bivariate_func_names():
                if self.output_shape_ == X.shape[0] * (X.shape[0] - 1) // 2:
                    include_diag = False
                elif self.output_shape_ == X.shape[0] * (X.shape[0] + 1) // 2:
                    include_diag = True
                self.feature_names_ = [
                    f'ch{i}-ch{j}' for _, i, j, in _idxiter(
                        X.shape[0], include_diag=include_diag)]

        return X_out

    def fit(self, X, y=None):
        """Fit the FeatureFunctionTransformer (does not extract features).

        Parameters
        ----------
        X : ndarray, shape (n_channels, n_times)

        y : ignored

        Returns
        -------
        self
        """
        self._check_input(X, reset=True)
        _feature_func = _get_python_func(self.func)
        if hasattr(_feature_func, 'get_feature_names'):
            _params = self.get_params()
            self.feature_names_ = _feature_func.get_feature_names(X, **_params)
        return self

    def get_feature_names(self):
        """Mapping of the feature indices to feature names."""
        if not hasattr(self, 'output_shape_'):
            raise ValueError('Call `fit_transform` first.')
        elif hasattr(self, 'feature_names_'):
            return self.feature_names_
        else:
            return np.arange(self.output_shape_).astype(str)

    def get_feature_names_out(self, input_features=None):
        """Mapping of the feature indices to feature names."""
        return self.get_feature_names()

    def get_params(self, deep=True):
        """Get the parameters (if any) of the given feature function.

        Parameters
        ----------
        deep : bool (default: True)
            If True, the method will get the parameters of the transformer.
            (See :class:`~sklearn.preprocessing.FunctionTransformer`).
        """
        func_to_inspect = _get_python_func(self.func)
        # Get code object from the function
        if hasattr(func_to_inspect, 'func_code'):
            func_code = func_to_inspect.func_code
        else:
            func_code = func_to_inspect.__code__
        args, _, _ = getargs(func_code)
        # Get defaults from the function
        if hasattr(func_to_inspect, 'defaults'):
            defaults = func_to_inspect.func_defaults
        else:
            defaults = func_to_inspect.__defaults__
        if defaults is None:
            return dict()
        else:
            n_defaults = len(defaults)
            func_params = {key: value for key, value in
                           zip(args[-n_defaults:], defaults)}
        if self.params is not None:
            func_params.update(self.params)
        return func_params

    def set_params(self, **new_params):
        """Set the parameters (if any) of the given feature function."""
        valid_params = self.get_params()
        for key in new_params.keys():
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for transformer %s. '
                                 'Check the list of available parameters '
                                 'using the `get_params` method of the '
                                 'transformer.' % (key, self))
        if self.params is not None:
            self.params.update(new_params)
        else:
            self.params = new_params
        self.kw_args = self.params
        return self


def _format_as_dataframe(X, feature_names):
    """Format to Pandas DataFrame.

    Utility function to format extracted features (X) as a Pandas
    DataFrame using names and indexes from ``feature_names``. The index of the
    columns is a MultiIndex with two levels. At level 0, the alias of the
    feature function is given. At level 1, an enumeration of the features is
    given.

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_features)
        Extracted features. X should be the output of :func:`extract_features`.

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
        _idx = [n.split('__')[1] for n in feature_names]
        _names = [n.split('__')[0] for n in feature_names]
        columns = pd.MultiIndex.from_arrays([_names, _idx])
        return pd.DataFrame(data=X, columns=columns)


def _apply_extractor(extractor, X, ch_names, return_as_df):
    """Utility function to apply features extractor to ndarray X.

    Parameters
    ----------
    extractor : Instance of :class:`~sklearn.pipeline.FeatureUnion` or
    :class:`~sklearn.pipeline.Pipeline`.

    X : ndarray, shape (n_channels, n_times)

    ch_names : list of str or None

    return_as_df : bool

    Returns
    -------
    X : ndarray, shape (n_features,)

    feature_names : list of str | None
        Not None, only if ``return_as_df`` is True.
    """
    X = extractor.fit_transform(X)
    feature_names = None
    if return_as_df:
        feature_names = extractor.get_feature_names_out()
        if ch_names is not None:  # rename channels
            mapping = {'ch%s' % i: ch_name
                       for i, ch_name in enumerate(ch_names)}
            for pattern, translation in mapping.items():
                r = re.compile(rf'{pattern}(?=_)|{pattern}\b')
                feature_names = [
                    r.sub(string=feature_name, repl=translation)
                    for feature_name in feature_names]

    return X, feature_names


def _check_funcs(selected, feature_funcs):
    """Selection checker.

    Checks if the elements of ``selected`` are either strings (alias of a
    feature function defined in mne-features) or tuples of the form
    ``(str, callable)`` (user-defined feature function).

    Parameters
    ----------
    selected : list of str or tuples
        Names of the selected feature functions.

    feature_funcs : dict
        Dictionary of the feature functions (univariate and bivariate)
        available in mne-features.

    Returns
    -------
    valid_funcs : list of tuples
    """
    valid_funcs = list()
    _intrinsic_func_names = feature_funcs.keys()
    for s in selected:
        if isinstance(s, str):
            # Case of a MNE-feature alias
            if s in _intrinsic_func_names:
                valid_funcs.append((s, feature_funcs[s]))
            else:
                raise ValueError('The given alias (%s) is not valid. The '
                                 'valid aliases for feature functions are: %s.'
                                 % (s, _intrinsic_func_names))
        elif isinstance(s, tuple):
            if len(s) != 2:
                raise ValueError('The given tuple (%s) is not of length 2. '
                                 'Each user-defined feature function should '
                                 'be passed as a tuple of the form '
                                 '`(str, callable)`.' % str(s))
            else:
                # Case of a user-defined feature function
                if s[0] in _intrinsic_func_names:
                    raise ValueError('A user-defined feature function was '
                                     'given an alias (%s) which is already '
                                     'used by mne-features. The list of '
                                     'aliases used by mne-features is: %s.'
                                     % (s[0], _intrinsic_func_names))
                else:
                    valid_funcs.append(s)
        else:
            # Case where the element is neither a string, nor a tuple
            raise ValueError('%s is not a valid feature function and cannot '
                             'be interpreted as a user-defined feature '
                             'function.' % str(s))
    if not valid_funcs:
        raise ValueError('No valid feature function was given.')
    else:
        return valid_funcs


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Feature extraction from epoched EEG data.

    The method ``fit_transform`` implemented in this class can be used to
    extract univariate or bivariate features from epoched data
    (see example below). The method ``fit`` is implemented for compatibility
    with Scikit-learn's API and extracts the feature names with format
    ``<ch_name>__<func_params>__<feature>``. As a result, the class
    ``FeatureExtractor`` can be used as a step in a Pipeline (see
    :class:`~sklearn.pipeline.Pipeline` and MNE-features examples). The class
    also accepts a ``memory`` parameter which allows for caching the result of
    feature extraction. Therefore, if caching is used, calling
    ``fit_transform`` twice on the same data will not trigger a second call
    to :func:`extract_features`.

    Parameters
    ----------
    sfreq : float (default: 256.)
        Sampling rate of the data.

    selected_funcs : list of str or tuples
        The elements of ``selected_features`` are either strings or tuples of
        the form ``(str, callable)``. If an element is of type ``str``, it is
        the alias of a feature function. The aliases are built from the
        feature functions' names by removing ``compute_``. For instance, the
        alias of the feature function :func:`compute_ptp_amp` is ``ptp_amp``.
        (See the documentation of mne-features). If an element is of type
        ``tuple``, the first element of the tuple should be a string
        (name/alias given to a user-defined feature function) and the second
        element should be a  callable (a user-defined feature function which
        accepts Numpy arrays with shape ``(n_channels, n_times)``). The
        names/aliases given to user-defined feature functions should not
        intersect the aliases used by mne-features. If the name given to a
        user-defined feature function is already used as an alias in
        mne-features, an error will be raised.

    params : dict or None (default: None)
        If not None, dict of optional parameters to be passed to
        :func:`extract_features`. Each key of the ``funcs_params`` dict should
        be of the form: ``[alias_feature_function]__[optional_param]``
        (for example: ``higuchi_fd__kmax``).

    ch_names : list of str or None (default: None)
        Channel names. Only used to get proper channels in ``feature_names``.
        If None, channel names will be of the form ``ch0``, ``ch1``, etc.

    n_jobs : int (default: 1)
        Number of CPU cores used when parallelizing the feature extraction.
        If given a value of -1, all cores are used.

    memory : str or None (default: None)
        If None, no caching is performed. If a string is given, the string
        should be the path to the caching directory. Caching is particularly
        advantageous when feature extraction is time consuming.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(42)
    >>> n_epochs, n_channels, n_times = 5, 3, 32
    >>> X = rng.randn(n_epochs, n_channels, n_times)
    >>> fe = FeatureExtractor(sfreq=100., selected_funcs=['std', 'kurtosis'])
    >>> X = fe.fit_transform(X)
    >>> print(X.shape)
    (5, 6)

    See also
    --------
    :func:`extract_features`
    """

    def __init__(self, sfreq=256., selected_funcs=None, params=None,
                 ch_names=None, n_jobs=1, memory=None):
        """Instantiate a FeatureExtractor object."""
        self.sfreq = sfreq
        self.selected_funcs = selected_funcs
        self.params = params
        self.ch_names = ch_names
        self.n_jobs = n_jobs
        self.memory = memory
        self.feature_names = None

    def fit(self, X, y=None):
        """Get the feature names in format
        ``<ch_name>__<func_params>__<feature>
        """
        # we get the channel names and create fake ones if containing '_'
        # (used later for parsing)
        unique_str = 'µ&mne-features&µ'
        if self.ch_names is None:
            tmp_ch_names = None
        else:
            tmp_ch_names = []
            for ch_name in self.ch_names:
                if unique_str in ch_name:
                    raise ValueError(
                        f'Channel name {ch_name} contains {unique_str}. '
                        f'Please use another name.')
                if '_' in ch_name:
                    # we replace with a special string that will be replaced
                    tmp_ch_names.append(ch_name.replace('_', unique_str))
                else:
                    tmp_ch_names.append(ch_name)

        # trick: use only the first epoch to get the feature names
        df = extract_features(
            X[:1],
            sfreq=self.sfreq,
            selected_funcs=self.selected_funcs,
            funcs_params=self.params,
            n_jobs=1,
            ch_names=tmp_ch_names,
            return_as_df=True,
        )
        cols = df.columns
        self.feature_names = [
            f"{cols[i][1].replace('_', '__').replace(unique_str, '_')}__"
            + f"{cols[i][0]}" for i in range(df.shape[1])
        ]
        return self

    def get_feature_names_out(self):
        """Get the feature names"""
        return self.feature_names

    def transform(self, X):
        """Extract features from the array X.

        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)

        Returns
        -------
        Xnew : ndarray, shape (n_epochs, n_features)
            Extracted features.
        """
        mem = joblib.Memory(location=self.memory)
        _extractor = mem.cache(extract_features)
        return _extractor(
            X,
            self.sfreq,
            self.selected_funcs,
            funcs_params=self.params,
            n_jobs=self.n_jobs,
        )

    def get_params(self, deep=True):
        """Get the parameters of the transformer."""
        return super(FeatureExtractor, self).get_params(deep=deep)

    def set_params(self, **params):
        """Set the parameters of the transformer."""
        self.params = params
        return self


def extract_features(X, sfreq, selected_funcs, funcs_params=None, n_jobs=1,
                     ch_names=None, return_as_df=False):
    """Extraction of temporal or spectral features from epoched EEG signals.

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_times)
        Array of epoched EEG data.

    sfreq : float
        Sampling rate of the data.

    selected_funcs : list of str or tuples
        The elements of ``selected_features`` are either strings or tuples of
        the form ``(str, callable)``. If an element is of type ``str``, it is
        the alias of a feature function. The aliases are built from the
        feature functions' names by removing ``compute_``. For instance, the
        alias of the feature function :func:`compute_ptp_amp` is ``ptp_amp``.
        (See the documentation of mne-features). If an element is of type
        ``tuple``, the first element of the tuple should be a string
        (name/alias given to a user-defined feature function) and the second
        element should be a  callable (a user-defined feature function which
        accepts Numpy arrays with shape ``(n_channels, n_times)``). The
        names/aliases given to user-defined feature functions should not
        intersect the aliases used by mne-features. If the name given to a
        user-defined feature function is already used as an alias in
        mne-features, an error will be raised.

    funcs_params : dict or None (default: None)
        If not None, dict of optional parameters to be passed to the feature
        functions. Each key of the ``funcs_params`` dict should be of the form:
        ``[alias_feature_function]__[optional_param]`` (for example:
        ``higuchi_fd__kmax``).

    n_jobs : int (default: 1)
        Number of CPU cores used when parallelizing the feature extraction.
        If given a value of -1, all cores are used.

    ch_names : list of str or None (default: None)
        If not None, list containing the names of each input channel.

    return_as_df : bool (default: False)
        If True, the extracted features will be returned as a Pandas DataFrame.
        The column index is a MultiIndex (see :class:`~pandas.MultiIndex`)
        which contains the alias of each feature function which was used.
        If False, the features are returned as a 2d Numpy array.

    Returns
    -------
    array-like, shape (n_epochs, n_features)
    """
    if sfreq <= 0:
        raise ValueError('Sampling rate `sfreq` must be positive.')
    univariate_funcs = get_univariate_funcs(sfreq)
    bivariate_funcs = get_bivariate_funcs(sfreq)
    feature_funcs = univariate_funcs.copy()
    feature_funcs.update(bivariate_funcs)
    sel_funcs = _check_funcs(selected_funcs, feature_funcs)

    if ch_names is not None and len(ch_names) != X.shape[1]:
        raise ValueError('`ch_names` should be of length {%s}' % X.shape[1])

    # Feature extraction
    n_epochs = X.shape[0]
    _tr = [(n, FeatureFunctionTransformer(func=func)) for n, func in sel_funcs]
    extractor = FeatureUnion(transformer_list=_tr)
    if funcs_params is not None:
        extractor.set_params(**funcs_params)
    res = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_apply_extractor)(
        extractor, X[j, :, :], ch_names, return_as_df) for j in range(n_epochs)
    )
    feature_names = res[0][1]
    res = list(zip(*res))[0]
    Xnew = np.vstack(res)
    if return_as_df:
        return _format_as_dataframe(Xnew, feature_names)
    else:
        return Xnew
