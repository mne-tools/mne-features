"""
================================
Analyze the feature importances.
================================

The example shows how feature names are forwarded in sklearn pipelines
so that one can have easy access to them for analysis purpose.

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
#         Etienne de Montalivet <etienne.demontalivet@protonmail.com>
# License: BSD 3 clause

import numpy as np
import mne
from mne.datasets import sample

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
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
# Prepare for the classification task
# -----------------------------------
#
# In addition to the new feature function, we also propose to extract the
# mean of the data:
selected_funcs = ["pow_freq_bands"]
params = {
    "pow_freq_bands__freq_bands": [0.5, 4, 8, 12, 30, 50, 70],
    "pow_freq_bands__log": True,
}

pipe = Pipeline(
    [
        (
            "fe",
            FeatureExtractor(
                sfreq=raw.info["sfreq"],
                ch_names=epochs.info["ch_names"],
                selected_funcs=selected_funcs,
                params=params,
            ).set_output(transform="pandas"),
        ),
        ("scaler", StandardScaler().set_output(transform="pandas")),
        ("clf", ExtraTreesClassifier(random_state=42)),
    ]
)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
y = labels

###############################################################################
# Print the accuracy score on a test dataset.

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
accuracy = pipe.fit(X_train, y_train).score(X_test, y_test)
print("Accuracy score = %1.3f" % accuracy)

###############################################################################
# Print the 10 most important features and their contribution.

best_10_id = np.argsort(pipe.named_steps["clf"].feature_importances_)[-10:][::-1]
best_10 = pipe.named_steps["clf"].feature_names_in_[best_10_id]
print(
    f"These 10 features ({best_10}) are responsible for "
    + f"{100*np.sum(pipe.named_steps['clf'].feature_importances_[best_10_id]):.2f}% "
    + "of the prediction."
)

###############################################################################
# Print the channel with most contribution.

feat_as_ch = np.array(
    [f.split("__")[0] for f in pipe.named_steps["clf"].feature_names_in_])
ch_names = epochs.info["ch_names"]
d = {ch: np.sum(
    pipe.named_steps["clf"].feature_importances_[feat_as_ch == ch]) for ch in ch_names}
best_ch = max(d, key=d.get)
print(f"Channel {best_ch} contributes the most with {100*d[best_ch]:.2f}%.")
