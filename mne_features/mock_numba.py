# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

"""Utility file to mock Numba and let the code work on any system."""

import sys
from warnings import warn

from mock import MagicMock

try:
    import numba as nb
except ImportError as _:
    warn('Numba not found. Your code will be slower.')

    sys.modules['numba'] = MagicMock()
    import numba as nb

    def jit(*args, **kwargs):
        def identity(ob):
            return ob
        return identity

    nb.jit = jit
