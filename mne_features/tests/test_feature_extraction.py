# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause


from tempfile import mkdtemp

import pytest
import numpy as np
from numpy.testing import assert_equal, assert_raises, assert_almost_equal
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils._mocking import CheckingClassifier
from mne_features.feature_extraction import (extract_features,
                                             FeatureFunctionTransformer,
                                             FeatureExtractor)
from mne_features.mock_numba import nb
from mne_features.univariate import compute_svd_fisher_info
from mne_features.utils import _idxiter


rng = np.random.RandomState(42)
sfreq = 256.
data = rng.standard_normal((10, 20, int(sfreq)))
n_epochs, n_channels = data.shape[:2]


def test_shape_output():
    sel_funcs = ['mean', 'variance', 'kurtosis', 'pow_freq_bands']
    features = extract_features(data, sfreq, sel_funcs, n_jobs=1)
    features_as_df = extract_features(data, sfreq, sel_funcs,
                                      n_jobs=1, return_as_df=True)
    expected_shape = (n_epochs, (3 + 5) * n_channels)
    assert_equal(features.shape, expected_shape)
    assert_equal(features, features_as_df.values)


def test_njobs():
    sel_funcs = ['app_entropy']
    features = extract_features(data, sfreq, sel_funcs, n_jobs=-1)
    expected_shape = (n_epochs, n_channels)
    assert_equal(features.shape, expected_shape)


def test_optional_params():
    features1 = extract_features(data, sfreq, ['spect_edge_freq'],
                                 {'spect_edge_freq__edge': [0.6]})
    features2 = extract_features(data, sfreq, ['spect_edge_freq'],
                                 {'spect_edge_freq__edge': [0.5, 0.95]})
    features3 = extract_features(data, sfreq, ['svd_fisher_info'],
                                 {'svd_fisher_info__tau': 5})
    assert_equal(features1.shape[-1], n_channels)
    assert_equal(features3.shape[-1], n_channels)
    assert_equal(features2.shape[-1], features1.shape[-1] * 2)


def test_optional_params_func_with_numba():
    sel_funcs = ['higuchi_fd']
    features1 = extract_features(data, sfreq, sel_funcs,
                                 {'higuchi_fd__kmax': 5})
    n_features1 = features1.shape[-1]
    assert_equal(n_features1, n_channels)


def test_wrong_params():
    with assert_raises(ValueError):
        # Negative sfreq
        extract_features(data, -0.1, ['mean'])
    with assert_raises(ValueError):
        # Unknown alias of feature function
        extract_features(data, sfreq, ['power_freq_bands'])
    with assert_raises(ValueError):
        # No alias given
        extract_features(data, sfreq, list())
    with assert_raises(ValueError):
        # Passing optional arguments with unknown alias
        extract_features(data, sfreq, ['higuchi_fd'],
                         {'higuch_fd__kmax': 3})


def test_featurefunctiontransformer():
    tr = FeatureFunctionTransformer(func=compute_svd_fisher_info)
    assert_equal(tr.get_params(), {'tau': 2, 'emb': 10})
    new_params = {'tau': 10}
    tr.set_params(**new_params)
    assert_equal(tr.get_params(), {'tau': 10, 'emb': 10})
    tr2 = FeatureFunctionTransformer(func=compute_svd_fisher_info,
                                     params={'emb': 20})
    assert_equal(tr2.get_params(), {'tau': 2, 'emb': 20})
    tr2.set_params(**new_params)
    assert_equal(tr2.get_params(), {'tau': 10, 'emb': 20})
    with assert_raises(ValueError):
        invalid_new_params = {'fisher_info_tau': 2}
        tr2.set_params(**invalid_new_params)


def test_feature_extractor():
    selected_funcs = ['app_entropy']
    extractor = FeatureExtractor(sfreq=sfreq, selected_funcs=selected_funcs)
    expected_features = extract_features(data, sfreq, selected_funcs)
    assert_almost_equal(expected_features, extractor.fit_transform(data))
    with assert_raises(ValueError):
        FeatureExtractor(
            sfreq=sfreq, selected_funcs=selected_funcs,
            params={'app_entropy__metric': 'sqeuclidean'}).fit_transform(data)


def test_gridsearch_feature_extractor():
    X = data
    y = np.ones((X.shape[0],))  # dummy labels
    pipe = Pipeline([('FE', FeatureExtractor(sfreq=sfreq,
                                             selected_funcs=['higuchi_fd'])),
                     ('clf', CheckingClassifier(
                         check_X=lambda arr: arr.shape[1:] == (X.shape[1],)))])
    params_grid = {'FE__higuchi_fd__kmax': [5, 10]}
    gs = GridSearchCV(estimator=pipe, param_grid=params_grid, cv=3)
    gs.fit(X, y)
    assert_equal(hasattr(gs, 'cv_results_'), True)


def test_memory_feature_extractor():
    selected_funcs = ['mean', 'zero_crossings']
    cachedir = mkdtemp()
    extractor = FeatureExtractor(sfreq=sfreq, selected_funcs=selected_funcs)
    cached_extractor = FeatureExtractor(sfreq=sfreq,
                                        selected_funcs=selected_funcs,
                                        memory=cachedir)
    y = np.ones((data.shape[0],))
    cached_extractor.fit_transform(data, y)
    # Ensure that the right features were cached
    assert_almost_equal(extractor.fit_transform(data, y),
                        cached_extractor.fit_transform(data, y))


def test_user_defined_feature_function():
    # User-defined feature function
    @nb.jit(nopython=True)
    def top_feature(arr, gamma=3.14):
        return np.sum(np.power(gamma * arr, 3) - np.power(arr / gamma, 2),
                      axis=-1)
    # Valid feature extraction
    selected_funcs = ['mean', ('top_feature', top_feature)]
    feat = extract_features(data, sfreq, selected_funcs)
    assert_equal(feat.shape, (n_epochs, 2 * n_channels))
    # Changing optional parameter ``gamma`` of ``top_feature``
    feat2 = extract_features(data, sfreq, selected_funcs,
                             funcs_params={'top_feature__gamma': 1.41})
    assert_equal(feat2.shape, (n_epochs, 2 * n_channels))
    # Invalid feature extractions
    with assert_raises(ValueError):
        # Alias is already used
        extract_features(data, sfreq, ['variance', ('mean', top_feature)])
        # Tuple is not of length 2
        extract_features(data, sfreq, ['variance', ('top_feature', top_feature,
                                                    data[:, ::2])])
        # Invalid type
        extract_features(data, sfreq, ['mean', top_feature])


def test_channel_naming():
    ch_names = ['CHANNEL%s' % i for i in range(n_channels)]
    ch_names[:4] = ['Cz', 'FCz', 'P1', 'CP1']
    selected_funcs = ['app_entropy']
    df = extract_features(
        data, sfreq, selected_funcs, ch_names=ch_names, return_as_df=True)
    expected_col_names = [('app_entropy', ch_name) for ch_name in ch_names]
    assert df.columns.values.tolist() == expected_col_names

    ch_names.append('CHANNEL%s' % n_channels)
    with assert_raises(ValueError):
        # incorrect number of channel names
        df = extract_features(
            data, sfreq, selected_funcs, ch_names=ch_names, return_as_df=True)


def test_channel_naming_pow_freq_bands():
    ch_names = ['CHANNEL%s' % i for i in range(n_channels)]
    ch_names[:4] = ['Cz', 'FCz', 'P1', 'CP1']
    selected_funcs = ['pow_freq_bands']
    func_params = {
        'pow_freq_bands__freq_bands': np.array([[0, 2], [10, 20]]),
        'pow_freq_bands__ratios': 'only'
    }
    df = extract_features(
        data, sfreq, selected_funcs, func_params, ch_names=ch_names,
        return_as_df=True)

    expected_col_names = [
        ('pow_freq_bands', f'{ch_name}_band{i}/band{j}')
        for ch_name in ch_names for _, i, j in _idxiter(2, triu=False)]
    assert df.columns.values.tolist() == expected_col_names


@pytest.mark.parametrize('selected_func', ['max_cross_corr', 'phase_lock_val',
                                           'nonlin_interdep'])
@pytest.mark.parametrize('include_diag', [True, False])
def test_channel_naming_bivariate(selected_func, include_diag):
    ch_names = ['CHANNEL%s' % i for i in range(n_channels)]
    ch_names[:4] = ['Cz', 'FCz', 'P1', 'CP1']
    func_params = {selected_func + '__include_diag': include_diag}
    df = extract_features(
        data, sfreq, [selected_func], func_params, ch_names=ch_names,
        return_as_df=True)
    expected_col_names = [
        (selected_func, ch_names[i] + '-' + ch_names[j])
        for s, i, j in _idxiter(n_channels, include_diag=include_diag)]

    assert df.columns.values.tolist() == expected_col_names


@pytest.mark.parametrize('selected_func', ['time_corr', 'spect_corr'])
@pytest.mark.parametrize('include_diag', [True, False])
@pytest.mark.parametrize('with_eig', [True, False])
def test_channel_naming_bivariate_eig(selected_func, include_diag, with_eig):
    ch_names = ['CHANNEL%s' % i for i in range(n_channels)]
    ch_names[:4] = ['Cz', 'FCz', 'P1', 'CP1']
    func_params = {
        selected_func + '__include_diag': include_diag,
        selected_func + '__with_eigenvalues': with_eig
    }
    df = extract_features(
        data, sfreq, [selected_func], func_params, ch_names=ch_names,
        return_as_df=True)
    expected_col_names = [
        (selected_func, ch_names[i] + '-' + ch_names[j])
        for s, i, j in _idxiter(n_channels, include_diag=include_diag)]
    if with_eig:
        expected_col_names.extend(
            [(selected_func, f'eig{i}') for i in range(n_channels)])

    assert df.columns.values.tolist() == expected_col_names


if __name__ == '__main__':

    test_shape_output()
    test_njobs()
    test_optional_params()
    test_optional_params_func_with_numba()
    test_wrong_params()
    test_featurefunctiontransformer()
    test_feature_extractor()
    test_gridsearch_feature_extractor()
    test_memory_feature_extractor()
    test_user_defined_feature_function()
    test_channel_naming()
    test_channel_naming_pow_freq_bands()
    test_channel_naming_bivariate()
