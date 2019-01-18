#!/usr/bin/env python
# -*- coding: utf-8 -*-
from codecs import open
from setuptools import setup, find_packages

import versioneer

# Get the long description from the README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

DISTNAME = 'ramp-backend'
DESCRIPTION = "Toolkit for interacting with the RAMP database."
MAINTAINER = 'Alexandre Boucaud'
MAINTAINER_EMAIL = 'boucaud.alexandre@gmail.com'
URL = 'https://github.com/paris-saclay-cds/ramp-backend'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/paris-saclay-cds/ramp-backend'
INSTALL_REQUIRES = ['click', 'numpy', 'psycopg2', 'sqlalchemy', 'six']


if __name__ == "__main__":
    setup(
        name=DISTNAME,
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
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
            'Operating System :: MacOS'],
        install_requires=INSTALL_REQUIRES,
        platforms='any',
        packages=find_packages(),
        entry_points={
            'console_scripts': ['ramp-launch = rampbkd.cli:start']
        }
    )