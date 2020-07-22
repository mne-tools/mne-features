MNE-Features
=========================================

|Travis|_ |Codecov|_

.. |Travis| image:: https://api.travis-ci.org/mne-tools/mne-features.svg?branch=master
.. _Travis: https://travis-ci.org/mne-tools/mne-features

.. |Codecov| image:: http://codecov.io/github/mne-tools/mne-features/coverage.svg?branch=master
.. _Codecov: http://codecov.io/github/mne-tools/mne-features?branch=master

This repository provides code for feature extraction with M/EEG data.
The documentation of the MNE-Features module is available at: `documentation <https://mne-tools.github.io/mne-features/index.html>`_.

Installation
------------

To install the package, the simplest way is to use pip to get the latest release::

  $ pip install mne-features

or to get the latest version of the code::

  $ pip install git+https://github.com/mne-tools/mne-features.git#egg=mne_features


Dependencies
------------

These are the dependencies to use MNE-Features:

* numpy (>=1.15.4)
* matplotlib (>=1.5)
* scipy (>=1.0)
* numba (>=0.45.1)
* scikit-learn (>=0.19)
* mne (>=0.18.2)
* PyWavelets (>=0.5.2)
* pandas (>=0.25.3)


Cite
----

If you use this code in your project, please cite::

    Jean-Baptiste SCHIRATTI, Jean-Eudes LE DOUGET, Michel LE VAN QUYEN, Slim ESSID, Alexandre GRAMFORT,
    "An ensemble learning approach to detect epileptic seizures from long intracranial EEG recordings"
    Proc. IEEE ICASSP Conf. 2018
