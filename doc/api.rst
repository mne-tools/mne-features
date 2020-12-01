.. _api_documentation:

=================
API Documentation
=================

.. currentmodule:: mne_features

Feature extraction
==================

Functions

.. currentmodule:: mne_features.feature_extraction

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: class.rst

   FeatureExtractor
   :template: function.rst
   extract_features

Univariate features
===================

Functions

.. currentmodule:: mne_features.univariate

.. autosummary::
   :toctree: generated/
   :template: function.rst

   get_univariate_funcs
   compute_mean
   compute_variance
   compute_std
   compute_ptp_amp
   compute_skewness
   compute_kurtosis
   compute_rms
   compute_prct
   compute_hurst_exp
   compute_app_entropy
   compute_samp_entropy
   compute_decorr_time
   compute_pow_freq_bands
   compute_hjorth_mobility_spect
   compute_hjorth_complexity_spect
   compute_hjorth_mobility
   compute_hjorth_complexity
   compute_higuchi_fd
   compute_katz_fd
   compute_zero_crossings
   compute_line_length
   compute_spect_slope
   compute_spect_entropy
   compute_svd_entropy
   compute_svd_fisher_info
   compute_energy_freq_bands
   compute_spect_edge_freq
   compute_wavelet_coef_energy
   compute_teager_kaiser_energy


Bivariate features
==================

Functions

.. currentmodule:: mne_features.bivariate

.. autosummary::
   :toctree: generated/

   get_bivariate_funcs
   compute_max_cross_corr
   compute_phase_lock_val
   compute_nonlin_interdep
   compute_time_corr
   compute_spect_corr
