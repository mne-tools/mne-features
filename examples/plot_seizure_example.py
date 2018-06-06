"""
============================================
Seizure detection example with MNE-Features.
============================================

The example is aimed at showing how MNE-Features can be used to design an
efficient seizure detection algorithm. To this end, the open Bonn iEEG dataset
is used. The dataset which is used in this example consists in 300 iEEG samples
(200 are seizure-free segments and 100 correspond to ictal activity). The data
was recorded at 173.61Hz on a single channel.

Some of the features used in this example were used in:

Jean-Baptiste SCHIRATTI, Jean-Eudes LE DOUGET, Michel LE VAN QUYEN,
Slim ESSID, Alexandre GRAMFORT,
"An ensemble learning approach to detect epileptic seizures from long
intracranial EEG recordings"
Proc. IEEE ICASSP Conf. 2018

.. note::

    This example is for illustration purposes, as other methods
    may lead to better performance on such a dataset (classification of
    "seizure" vs. "non-seizure" iEEG segments). For further information, see
    (Andrzejak et al., 2001) and http://epileptologie-bonn.de.
"""

# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

import os
import os.path as op

from download import download

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from mne_features.feature_extraction import FeatureExtractor

print(__doc__)

###############################################################################
# Let us download the iEEG data from the Bonn dataset:


def download_bonn_ieeg(path, verbose=False):
    base_url = 'http://epileptologie-bonn.de/cms/upload/workgroup/lehnertz/'
    urls = [('setC', 'N.zip'), ('setD', 'F.zip'), ('setE', 'S.zip')]
    paths = list()
    for set_name, url_suffix in urls:
        _path = download(op.join(base_url, url_suffix),
                         op.join(path, set_name), kind='zip', replace=False,
                         verbose=verbose)
        paths.append(_path)
    return paths


# Download the data to ``./bonn_data``:
paths = download_bonn_ieeg('./bonn_data')

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

# Shape of extracted data:
print(data.shape)

###############################################################################
# Prepare for the classification task:
selected_funcs = ['line_length', 'kurtosis', 'ptp_amp', 'skewness']

pipe = Pipeline([('fe', FeatureExtractor(sfreq=sfreq,
                                         selected_funcs=selected_funcs)),
                 ('clf', RandomForestClassifier(n_estimators=100,
                                                max_depth=4,
                                                random_state=42))])
skf = StratifiedKFold(n_splits=3, random_state=42)

###############################################################################
# Print the cross-validation accuracy score:

scores = cross_val_score(pipe, data, y, cv=skf)
print('Cross-validation accuracy score = %1.3f (+/- %1.5f)' % (np.mean(scores),
                                                               np.std(scores)))
