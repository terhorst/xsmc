import glob

import numpy as np
from setuptools import find_packages, setup
from setuptools.extension import Extension

from numpy.distutils.misc_util import get_info
npymath_info = get_info('npymath')

USE_CYTHON = True  # os.environ.get("USE_CYTHON", True)

ext = ".pyx" if USE_CYTHON else ".cpp"

tskit_sourcefiles = ["tskit/c/subprojects/kastore/kastore.c"] + list(
    glob.glob("tskit/c/tskit/*.c")
)

include_dirs = ["include", "tskit/c", "tskit/c/subprojects/kastore", np.get_include()]
npymath_info.setdefault('include_dirs', [])
npymath_info['include_dirs'] += include_dirs

extensions = [
    Extension(
        "xsmc._tskit",
        ["tskit/python/_tskitmodule.c"] + tskit_sourcefiles,
        include_dirs=include_dirs,
    ),
    Extension(
        "xsmc._viterbi",
        ["xsmc/_viterbi.pyx"] + tskit_sourcefiles,
        language="c++",
        include_dirs=include_dirs,
    ),
    Extension(
        "xsmc._sampler",
        ["xsmc/_sampler.pyx"] + tskit_sourcefiles,
        language="c++",
        **npymath_info,
        extra_compile_args=["-Wfatal-errors"]
    ),
]

setup(
    name="xsmc",
    author="Caleb Ki and Jonathan Terhorst",
    author_email="jonth@umich.edu",
    install_requires=[
        "numpy>=1.17.0",
        "scipy>=1.5.0",
        "matplotlib>=3.0.0",
        "tskit>=0.3.1",
    ],
    packages=find_packages(),
    ext_modules=extensions,
)
