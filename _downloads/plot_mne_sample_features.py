"""
========================================================
Extract features from MEG time series for classification
========================================================

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
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

import mne
import numpy as np
from mne.datasets import sample
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mne_features.feature_extraction import extract_features

print(__doc__)

###############################################################################
# Let us import the data using MNE-Python and epoch it:

data_path = sample.data_path()
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

pipe = Pipeline([('scaler', StandardScaler()),
                 ('lr', LogisticRegression(random_state=42))])
y = labels

###############################################################################
# Classification using features (mean, peak-to-peak amplitude,
# standard deviation). See :ref:`api_documentation` for full list of supported
# features.

selected_funcs = {'mean', 'ptp_amplitude', 'std'}
X_new = extract_features(data, raw.info['sfreq'], selected_funcs)
kf = KFold(n_splits=3, random_state=42)
scores = cross_val_score(pipe, X_new, y, scoring='accuracy', cv=kf)

###############################################################################
# Print the cross-validation score:

print('Cross-validation accuracy score = %1.3f (+/- %1.5f)' % (np.mean(scores),
                                                               np.std(scores)))
