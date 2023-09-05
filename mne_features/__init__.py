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

__version__ = '0.3'

from . import univariate, bivariate, feature_extraction
from .univariate import get_univariate_funcs
from .bivariate import get_bivariate_funcs
