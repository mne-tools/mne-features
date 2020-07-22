"""Mock Numba and let mne-features work on any system."""

# Author: Jean-Baptiste Schiratti <jean.baptiste.schiratti@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

try:
    import numba as nb
except ImportError as err:
    print('{}. Your code will be slower.'.format(err))

    class Bunch(dict):
        """Dictionnary-like object that exposes its keys as attributes."""

        def __init__(self, **kwargs):  # noqa: D102
            dict.__init__(self, kwargs)
            self.__dict__ = self

    class MockType(object):
        def __getitem__(self, slice):
            return self

        def __call__(self, *args, **kwargs):
            return

    nb = Bunch()
    nb.int32 = MockType()
    nb.int64 = MockType()
    nb.float32 = MockType()
    nb.float64 = MockType()
    nb.optional = MockType()
    nb.boolean = MockType()

    def jit(*args, **kwargs):
        def identity(ob):
            return ob
        return identity

    nb.jit = jit
