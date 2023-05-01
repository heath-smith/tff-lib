#!/usr/bin/env python
"""
This script provides a shim for setuptools which is needed
for local editable installations.
"""

from setuptools import setup, Extension
import numpy

setup(
    ext_modules=[
        Extension(
            name='medium',
            sources=['tff_lib/medium.cc'],
            include_dirs=[numpy.get_include()]
        ),
    ]
)
