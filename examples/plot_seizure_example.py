"""
============================================
Seizure detection example with MNE-Features.
============================================

The example is aimed at showing how MNE-Features can be used to design an
efficient seizure detection algorithm. To this end, the open Bonn iEEG dataset
is used. The dataset which is used in this example consists in 300 iEEG samples
(200 are seizure-free segments and 100 correspond to ictal activity). The data
was recorded at 173.61Hz on a single channel.

The code for this example is based on the method proposed in:

Jean-Baptiste SCHIRATTI, Jean-Eudes LE DOUGET, Michel LE VAN QUYEN,
Slim ESSID, Alexandre GRAMFORT,
"An ensemble learning approach to detect epileptic seizures from long
intracranial EEG recordings"
Proc. IEEE ICASSP Conf. 2018

.. note::

    This example is for illustration purposes, as other methods
    may lead to better performance on such a dataset (classification of
    "seizure" vs. "non-seizure" iEEG segments).

References
----------

.. [1] Andrzejak, R. G. et al. (2001). Indications of nonlinear deterministic
       and finite-dimensional structures in time series of brain electrical
       activity: Dependence on recording region and brain state. Physical
       Review E, 64(6), 061907.

.. [2] http://epileptologie-bonn.de/cms/front_content.php?idcat=193&lang=3
"""

# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

import os
import numpy as np
import os.path as op
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.pipeline import Pipeline

from mne_features.feature_extraction import FeatureExtractor
from mne_features.utils import download_bonn_ieeg
print(__doc__)

###############################################################################
# Let us download the iEEG data from the Bonn dataset:

# Download the data to ``./bonn_data``:
data_path = op.join(op.dirname(__file__), 'bonn_data')
paths = download_bonn_ieeg(data_path)

# Read the data from ``.txt`` files. Only the iEEG epochs in
# ``./bonn_data/setE`` correspond to ictal
# activity.
data_segments = list()
labels = list()
sfreq = 173.61
for path in paths:
    fnames = [s for s in os.listdir(path) if s.lower().endswith('.txt')]
    for fname in fnames:
        _data = pd.read_csv(op.join(path, fname), sep='\n', header=None)
        data_segments.append(_data.values.T[None, ...])
    if 'setE' in path:
        labels.append(np.ones((len(fnames),)))
    else:
        labels.append(np.zeros((len(fnames),)))
data = np.concatenate(data_segments)
y = np.concatenate(labels, axis=0)
os.removedirs(data_path)

# Shape of extracted data:
print(data.shape)

###############################################################################
# Prepare for the classification task:
selected_funcs = ['mean', 'line_length']

pipe = Pipeline([('fe', FeatureExtractor(sfreq=sfreq,
                                         selected_funcs=selected_funcs)),
                 ('clf', RandomForestClassifier(n_estimators=500,
                                                random_state=42))])
skf = StratifiedKFold(n_splits=5, random_state=42)

###############################################################################
# Print the cross-validation accuracy score:

scores = cross_val_score(pipe, data, y, cv=skf)
print('Cross-validation accuracy score = %1.3f (+/- %1.5f)' % (np.mean(scores),
                                                               np.std(scores)))
