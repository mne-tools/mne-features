"""
===================================================================
Estimation of the slope and intercept of the Power Spectral Density
===================================================================

This example aims at showing how the utility function `power_spectrum` and 
the feature function :func:`mne_features.univariate.compute_spect_slope` can 
be used to estimate the slope and the intercept of the Power Spectral 
Density (PSD, computed using the FFT of the signal).

The code for this example is based on the method proposed in:

Jean-Baptiste SCHIRATTI, Jean-Eudes LE DOUGET, Michel LE VAN QUYEN,
Slim ESSID, Alexandre GRAMFORT,
"An ensemble learning approach to detect epileptic seizures from long
intracranial EEG recordings"
Proc. IEEE ICASSP Conf. 2018

.. note::

    This example is for illustration purposes, as other methods
    may lead to more robust/reliable estimation of the slope and intercept 
    of the PSD.

"""  # noqa

# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import sample

from mne_features.univariate import compute_spect_slope
from mne_features.utils import power_spectrum

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

###############################################################################
# Estimate the slope (and the intercept) of the PSD. For the sake of the
# example, a single epoch and channel are used to estimate the slope and the
# intercept.

data = epochs.get_data()[0, 1, :].reshape((1, -1))
sfreq = epochs.info['sfreq']

# Compute the (one-sided) PSD using FFT. The ``mask`` variable allows to
# select only the part of the PSD which corresponds to frequencies between
# 0.1Hz and 40Hz (the data used in this example is already low-pass filtered
# at 40Hz).
_psd, _freqs = power_spectrum(sfreq, data)
mask = np.logical_and(0.1 <= _freqs, _freqs <= 40)

psd, freqs = _psd[0, mask], _freqs[mask]

# Estimate the slope (and the intercept) of the PSD. The function
# :func:`compute_spect_slope` assumes that the PSD of the signal is of the
# form: ``psd[f] = b / (f ** a)``. The coefficients a and b are respectively
# called *slope* and *intercept* of the Power Spectral Density. The values of
# the variables ``slope`` and ``intercept`` differ from the values returned
# by ``compute_spect_slope`` because, in the feature function, the linear
# regression fit is done in the log10-log10 scale.
res = compute_spect_slope(sfreq, data, fmin=0.1, fmax=40)
slope = -res[1]
intercept = 10 ** res[0]
print('The estimated slope (respectively intercept) is: %1.2f (resp. %1.3e)' %
      (slope, intercept))

# Plot the PSD together with the ``b / (f ** a)`` curve (estimated decay of
# the PSD with frequency) and the ``b / f`` curve (classical decay of the PSD
# with frequency).
plt.figure()
plt.plot(freqs, psd, '-b', lw=2, label='PSD')
plt.plot(freqs, intercept / (freqs ** slope), '-r', lw=2,
         label='b / (f ** a)')
plt.plot(freqs, intercept / (freqs ** 1.), '-m', lw=1, label='b / f')
plt.xlabel('f (frequency, in Hz)')
plt.ylabel('PSD (in Volts ** 2 / Hz)')
plt.ylim([np.min(psd), np.max(psd)])
plt.xlim([1, 40])
plt.legend(loc='upper right')
plt.show()
