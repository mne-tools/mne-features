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

   extract_features

Univariate features
===================

Functions

.. currentmodule:: mne_features.univariate

.. autosummary::
   :toctree: generated/

   get_univariate_funcs
   compute_mean
   compute_variance
   compute_std
   compute_ptp
   compute_skewness
   compute_kurtosis
   compute_hurst_exponent
   compute_app_entropy
   compute_samp_entropy
   compute_decorr_time
   power_spectrum
   compute_power_spectrum_freq_bands
   compute_spect_hjorth_mobility
   compute_spect_hjorth_complexity
   compute_hjorth_mobility
   compute_hjorth_complexity
   compute_higuchi_fd
   compute_katz_fd
   compute_zero_crossings
   compute_line_length
   compute_spect_entropy
   compute_svd_entropy
   compute_svd_fisher_info
   compute_energy_freq_bands
   compute_spect_edge_freq
   compute_wavelet_coef_energy

Bivariate features
==================

Functions

.. currentmodule:: mne_features.bivariate

.. autosummary::
   :toctree: generated/

   get_bivariate_funcs
   compute_max_cross_correlation
   compute_phase_locking_value
   compute_nonlinear_interdep
   compute_time_corr_coefs
   compute_spect_corr_coefs