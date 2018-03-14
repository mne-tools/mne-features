"""MNE-Features software for extracting features from multivariate time series."""  # noqa

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.devN' where N is an integer.
#

__version__ = '0.1.dev0'

from .univariate import (compute_mean, compute_variance, compute_std,
                         compute_ptp, compute_skewness, compute_kurtosis,
                         compute_hurst_exponent, compute_app_entropy,
                         compute_samp_entropy, compute_decorr_time,
                         power_spectrum, compute_power_spectrum_freq_bands,
                         compute_spect_hjorth_mobility,
                         compute_spect_hjorth_complexity,
                         compute_hjorth_mobility, compute_hjorth_complexity,
                         compute_higuchi_fd, compute_katz_fd)

from .bivariate import (compute_max_cross_correlation,
                        compute_nonlinear_interdep,
                        compute_phase_locking_value, compute_spect_corr_coefs,
                        compute_time_corr_coefs)
