.. _whats_new:

What's new?
===========

.. currentmodule:: mne_features

.. _current:

Current
-------

Changelog
~~~~~~~~~

Enhancements
~~~~~~~~~~~~

- Extended feature naming to all univariate functions. If no specific function is implemented for a given univariate function and that the output has the same shape as the input on the feature dimension, then a generic list of feature names is passed. Possibility to rename these feature by passing ``ch_names`` to :func:`mne_features.extract_features`. By `Paul ROUJANSKY`_ in `#60 <https://github.com/mne-tools/mne-features/pull/60>`_.
- Added root mean square (RMS) and percentile univariate feature extraction functions. By `Hubert Banville`_ in `#61 <https://github.com/mne-tools/mne-features/pull/61>`_.
- Added ``log`` and ``ratios_triu`` to :func:`mne_features.univariate.compute_pow_freq_bands` to allow computing log-power and log-ratios of power and controlling whether all possible band ratios should be computed.  By `Hubert Banville`_ in `#62 <https://github.com/mne-tools/mne-features/pull/62>`_.
- Extended feature naming to all bivariate functions, with the same option to pass ``ch_names``  to :func:`mne_features.extract_features` as for univariate functions. By `Hubert Banville`_ in `#63 <https://github.com/mne-tools/mne-features/pull/63>`_.

Bug
~~~

- Fixed the behavior of the `edge` parameter in :func:`mne_features.univariate.compute_spect_edge_freq`. Valid values for this parameter are now `None` or a list of float between `0` and `1` (percentages). By `Jean-Baptiste SCHIRATTI`_ in `#52 <https://github.com/mne-tools/mne-features/pull/52>`_.
- Fixed channel name replacement with overlapping channel names (e.g., Cz and FCz) which affected :func:`mne_features.extract_features` with `return_as_df=True` and when `ch_names` are provided. By `Hubert Banville`_ in `#63 <https://github.com/mne-tools/mne-features/pull/63>`_.

API
~~~

.. _Jean-Baptiste Schiratti: https://github.com/jbschiratti
.. _Alex Gramfort: http://alexandre.gramfort.net
.. _Paul Roujansky: https://github.com/paulroujansky
.. _Hubert Banville: https: https://hubertjb.github.io/
