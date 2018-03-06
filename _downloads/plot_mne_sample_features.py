"""
=======================================================
Extract features from MEG time series
=======================================================

The example is based on the algorithm proposed in:

Jean-Baptiste SCHIRATTI, Jean-Eudes LE DOUGET, Michel LE VAN QUYEN,
Slim ESSID, Alexandre GRAMFORT,
"An ensemble learning approach to detect epileptic seizures from long
intracranial EEG recordings"
Proc. IEEE ICASSP Conf. 2018

"""  # noqa

# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

import mne
from mne.datasets import sample
from mne_features import compute_kurtosis

print(__doc__)

###############################################################################
# Generate sample EEG data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.1, 0.4
event_id = dict(aud_l=1, vis_l=3)

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.filter(.5, None, fir_design='firwin')
events = mne.read_events(event_fname)
picks = mne.pick_types(raw.info, meg=True, eeg=False)

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks, proj=True,
                    decim=2, baseline=None, preload=True)
labels = epochs.events[:, -1]

# get MEG and EEG data
data = epochs.get_data()

###############################################################################
# Extract some features to classify epochs

X = compute_kurtosis(data)

###############################################################################
# Run classification

# XXX TODO
