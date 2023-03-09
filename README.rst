MNE-Features
=========================================

|GitHub Actions|_ |Codecov|_

.. |GitHub Actions| image:: https://github.com/mne-tools/mne-features/actions/workflows/main.yml/badge.svg
.. _GitHub Actions: https://github.com/mne-tools/mne-features/actions/workflows/main.yml

.. |Codecov| image:: http://codecov.io/github/mne-tools/mne-features/coverage.svg?branch=master
.. _Codecov: http://codecov.io/github/mne-tools/mne-features?branch=master

This repository provides code for feature extraction with M/EEG data.
The documentation of the MNE-Features module is available at: `documentation <https://mne-tools.github.io/mne-features/index.html>`_.

Installation
------------

To install the package, the simplest way is to use ``pip`` to get the latest release::

  $ pip install mne-features

Or if you prefer ``conda``::

  $ conda install --channel=conda-forge mne-features

Or to get the latest version of the code::

  $ pip install git+https://github.com/mne-tools/mne-features.git#egg=mne_features


Dependencies
------------

These are the dependencies to use MNE-Features:

* numpy (>=1.17)
* matplotlib (>=1.5)
* scipy (>=1.0)
* numba (>=0.46.0)
* llvmlite (>=0.30)
* scikit-learn (>=0.21)
* mne (>=0.18.2)
* PyWavelets (>=0.5.2)
* pandas (>=0.25)


Cite
----

If you use this code in your project, please cite::

    Jean-Baptiste SCHIRATTI, Jean-Eudes LE DOUGET, Michel LE VAN QUYEN, Slim ESSID, Alexandre GRAMFORT,
    "An ensemble learning approach to detect epileptic seizures from long intracranial EEG recordings"
    Proc. IEEE ICASSP Conf. 2018
