import os
import platform
import sys

import numpy as np
from setuptools import find_packages, setup
from setuptools.extension import Extension

USE_CYTHON = os.environ.get("USE_CYTHON", False)

ext = ".pyx" if USE_CYTHON else ".cpp"

tskit_sourcefiles = [
    "src/kastore.c",
    "src/tskit/core.c",
    "src/tskit/genotypes.c",
    "src/tskit/tables.c",
    "src/tskit/trees.c",
]

include_dirs = ["src/", np.get_include()]

extensions = [
    Extension(
        "xsmc._lwtc",
        ["src/_lwtc.c"],
        language="c",
        include_dirs=include_dirs,
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
        "tskit==0.3.1",
        "matplotlib>=3.0.0",
    ],
    setup_requires=["setuptools>=46.0", "cython>=0.29"],
    packages=find_packages(),
    ext_modules=extensions,
)
