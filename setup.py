#! /usr/bin/env python

import os
import setuptools  # noqa; we are using a setuptools namespace
from numpy.distutils.core import setup

descr = """MNE-Features software for extracting features from multivariate time series"""  # noqa

version = None
with open(os.path.join('mne_features', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


DISTNAME = 'python-mne_features'
DESCRIPTION = descr
MAINTAINER = 'Jean-Baptiste Schiratti'
MAINTAINER_EMAIL = 'jean.baptiste.schiratti@gmail.com'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/mne-tools/mne-features.git'
VERSION = version


def package_tree(pkgroot):
    """Get the submodule list."""
    # Adapted from VisPy
    path = os.path.dirname(__file__)
    subdirs = [os.path.relpath(i[0], path).replace(os.path.sep, '.')
               for i in os.walk(os.path.join(path, pkgroot))
               if '__init__.py' in i[2]]
    return sorted(subdirs)


if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.rst').read(),
          long_description_content_type='text/x-rst',
          classifiers=['Intended Audience :: Science/Research',
                       'Intended Audience :: Developers',
                       'License :: OSI Approved',
                       'Programming Language :: Python',
                       'Topic :: Software Development',
                       'Topic :: Scientific/Engineering',
                       'Operating System :: Microsoft :: Windows',
                       'Operating System :: POSIX',
                       'Operating System :: Unix',
                       'Operating System :: MacOS',
                       'Programming Language :: Python :: 3',
                       ],
          platforms='any',
          python_requires='>=3.6',
          packages=package_tree('mne_features'),
          install_requires=['numpy', 'scipy', 'numba', 'scikit-learn', 'mne',
                            'PyWavelets', 'pandas'],
          )
