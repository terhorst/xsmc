from setuptools import setup, find_packages
from setuptools.extension import Extension
import os
import sys
import platform
import glob


USE_CYTHON = os.environ.get("USE_CYTHON", True)

ext = ".pyx" if USE_CYTHON else ".cpp"

tskit_sourcefiles = [
    "tskit/c/subprojects/kastore/kastore.c",
] + list(glob.glob('tskit/c/tskit/*.c'))

include_dirs = ['include', "tskit/c", "tskit/c/subprojects/kastore"]

extensions = [
    Extension(
        'xsmc._tskit',
        ['tskit/python/_tskitmodule.c'] + tskit_sourcefiles,
        include_dirs = include_dirs
    ),
    Extension(
        "xsmc._viterbi",
        ["xsmc/_viterbi.pyx"] + tskit_sourcefiles,
        language="c++",
        libraries=["gsl", "gslcblas"],
        include_dirs=include_dirs,
    ),
    Extension(
        "xsmc._sampler",
        ["xsmc/_sampler.pyx"] + tskit_sourcefiles,
        language="c++",
        include_dirs=include_dirs,
        # extra_compile_args=["-Wfatal-errors"]
    ),
]

setup(
    name="xsmc",
    version="1.0.0",
    author="Caleb Ki and Jonathan Terhorst",
    author_email="jonth@umich.edu",
    install_requires=[
        "numpy>=1.17.0",
        "scipy>=1.1.0",
        "matplotlib>=3.0.0",
    ],
    setup_requires=["setuptools>=46.0", "cython>=0.29"],
    packages=find_packages(),
    ext_modules=extensions,
)
