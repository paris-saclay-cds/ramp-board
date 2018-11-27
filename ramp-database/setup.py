#!/usr/bin/env python
# -*- coding: utf-8 -*-
from codecs import open
from setuptools import setup, find_packages

version = '0.0.1'
# Get the long description from the README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

DISTNAME = 'ramp-database'
DESCRIPTION = "RAMP database model"
MAINTAINER = 'Alexandre Boucaud'
MAINTAINER_EMAIL = 'boucaud.alexandre@gmail.com'
URL = 'https://github.com/paris-saclay-cds/ramp-board'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/paris-saclay-cds/ramp-board'


if __name__ == "__main__":
    setup(
        name=DISTNAME,
        version=version,
        maintainer=MAINTAINER,
        include_package_data=True,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        download_url=DOWNLOAD_URL,
        long_description=long_description,
        zip_safe=False,  # the package can run out of an .egg file
        classifiers=[
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved',
            'Programming Language :: Python',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
        ],
        install_requires=[
            'numpy',
            'pyyaml',
            'sqlalchemy',
            'psycopg2',
        ],
        platforms='any',
        packages=find_packages(),
        )
