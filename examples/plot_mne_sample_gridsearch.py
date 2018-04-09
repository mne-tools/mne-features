"""
====================================================================
Optimized feature extraction from MEG time series for classification
====================================================================

The code for this example is based on the method proposed in:

Jean-Baptiste SCHIRATTI, Jean-Eudes LE DOUGET, Michel LE VAN QUYEN,
Slim ESSID, Alexandre GRAMFORT,
"An ensemble learning approach to detect epileptic seizures from long
intracranial EEG recordings"
Proc. IEEE ICASSP Conf. 2018

.. note::

    This example is for illustration purposes, as other methods
    may lead to better performance on such a dataset (classification
    of auditory vs. visual stimuli).

"""  # noqa

# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Guillaume Corda <guillaume.corda@telecom-paristech.fr>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

import mne
import numpy as np
from mne_features.feature_extraction import FeatureExtraction
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     StratifiedKFold)
from sklearn.pipeline import Pipeline

print(__doc__)

###############################################################################
# Let us import the data using MNE-Python and epoch it:

data_path = '/home/guillaume/Workspace/MNE-sample-data'
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.filter(.5, None, fir_design='firwin')
events = mne.read_events(event_fname)
picks = mne.pick_types(raw.info, meg='grad', eeg=False)

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks, proj=True,
                    baseline=None, preload=True)
labels = epochs.events[:, -1]

# get MEG and EEG data
data = epochs.get_data()

###############################################################################
# Prepare for the classification task:

pipe = Pipeline([('FE', FeatureExtraction(sfreq=raw.info['sfreq'],
                                          selected_funcs=['app_entropy',
                                                          'mean'])),
                 ('clf', LogisticRegression(random_state=42))])
skf = StratifiedKFold(n_splits=3, random_state=42)
y = labels

###############################################################################
# Cross-validation accuracy score with default parameters (emb = 2, by default
# for `compute_app_entropy`):

scores = cross_val_score(pipe, data, y, cv=skf)
print('Cross-validation accuracy score (with default parameters) = %1.3f '
      '(+/- %1.5f)' % (np.mean(scores), np.std(scores)))

###############################################################################
# Optimization of features extraction optional parameters:
# Here, only the embedding dimension parameter of `compute_app_entropy` is
# optimized using GridSearchCV.

params_grid = {'FE__app_entropy__emb': np.arange(2, 11)}

gs = GridSearchCV(estimator=pipe, param_grid=params_grid,
                  cv=StratifiedKFold(n_splits=2, random_state=52), n_jobs=-1)
gs.fit(data, y)

# Best parameters obtained with GridSearchCV:
print(gs.best_params_)

###############################################################################
# Cross-validation accuracy score with optimized parameters:

gs_best = gs.best_estimator_
new_scores = cross_val_score(gs_best, data, y, cv=skf)

print('Cross-validation accuracy score (with optimized parameters) = %1.3f '
      '(+/- %1.5f)' % (np.mean(new_scores), np.std(new_scores)))
