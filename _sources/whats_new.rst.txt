.. _whats_new:

What's new?
===========

.. currentmodule:: mne_features

.. _current:

Current
-------

Changelog
~~~~~~~~~

Bug
~~~

- Fixed the behavior of the `edge` parameter in :func:`mne_features.univariate.compute_spect_edge_freq`. Valid values for this parameter are now `None` or a list of float between `0` and `1` (percentages). By `Jean-Baptiste SCHIRATTI`_ in `#52 <https://github.com/mne-tools/mne-features/pull/52>`_.

API
~~~

.. _Jean-Baptiste Schiratti: https://github.com/jbschiratti
.. _Alex Gramfort: http://alexandre.gramfort.net