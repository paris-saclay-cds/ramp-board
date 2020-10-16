#! /usr/bin/env python
import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('ramp_frontend', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'ramp-frontend'
DESCRIPTION = "Website for RAMP"
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'A. Boucaud, B. Kegl, G. Lemaitre, J. Van den Bossche'
MAINTAINER_EMAIL = 'boucaud.alexandre@gmail.com, guillaume.lemaitre@inria.fr'
URL = 'https://github.com/paris-saclay-cds/ramp-board'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/paris-saclay-cds/ramp-board'
VERSION = __version__  # noqa
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8']
INSTALL_REQUIRES = ['bokeh', 'click', 'Flask', 'Flask-Login', 'Flask-Mail',
                    'Flask-SQLAlchemy', 'Flask-WTF', 'itsdangerous', 'numpy',
                    'pandas']
EXTRAS_REQUIRE = {
    'tests': ['pytest', 'pytest-cov'],
    'docs': ['sphinx', 'sphinx_rtd_theme', 'numpydoc']
}
PACKAGE_DATA = {
    'ramp_frontend': [
        os.path.join('templates', '*'),
        os.path.join('static', 'css', 'style.css'),
        os.path.join('static', 'css', 'themes', 'flat-blue.css'),
        os.path.join('static', 'img', '*'),
        os.path.join('static', 'img', 'backdrop', '*'),
        os.path.join('static', 'img', 'partners', '*'),
        os.path.join('static', 'img', 'powered_by', '*'),
        os.path.join('static', 'js', '*'),
        os.path.join('static', 'lib', 'css', '*'),
        os.path.join('static', 'lib', 'fonts', '*'),
        os.path.join('static', 'lib', 'img', '*'),
        os.path.join('static', 'lib', 'js', '*'),
    ]
}

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    package_data=PACKAGE_DATA,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        'console_scripts': ['ramp-frontend = ramp_frontend.cli:start']
    }
)
