#!/usr/bin/env python
"""
Minimal setup.py for versioneer compatibility.
All package configuration is now in pyproject.toml (PEP 621).
"""

from setuptools import setup

import versioneer

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
