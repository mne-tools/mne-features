"""
======================================================
Extract features using user-defined feature functions.
======================================================

The example shows how user-defined feature functions can be used in
MNE-Features along with built-in feature functions.

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

"""

# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

from scipy.signal import medfilt

import mne
from mne.datasets import sample

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split, StratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mne_features.feature_extraction import FeatureExtractor

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
picks = mne.pick_types(raw.info, meg=False, eeg=True)

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks, proj=True,
                    baseline=None, preload=True)
labels = epochs.events[:, -1]

# get MEG and EEG data
data = epochs.get_data()

###############################################################################
# Define a feature function called ``compute_medfilt``
# ----------------------------------------------------
#
# Here, the raw data is median filtered and the output signals are used
# as features.


def compute_medfilt(arr):
    """Median filtered signal as features.

    Parameters
    ----------
    arr : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : (n_channels * n_times,)
    """
    return medfilt(arr, kernel_size=(1, 5)).ravel()


###############################################################################
# Prepare for the classification task
# -----------------------------------
#
# In addition to the new feature function, we also propose to extract the
# mean of the data:
selected_funcs = [('medfilt', compute_medfilt), 'mean']

pipe = Pipeline([('fe', FeatureExtractor(sfreq=raw.info['sfreq'],
                                         selected_funcs=selected_funcs)),
                 ('scaler', StandardScaler()),
                 ('clf', LogisticRegression(random_state=42))])
skf = StratifiedKFold(n_splits=3, random_state=42)
y = labels

###############################################################################
# Print the accuracy score on a test dataset.

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
accuracy = pipe.fit(X_train, y_train).score(X_test, y_test)
print('Accuracy score = %1.3f' % accuracy)
