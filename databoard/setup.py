from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=[
        Extension("_isotonic", ["_isotonic.c"],
        include_dirs=[numpy.get_include()]),
    ],
)

