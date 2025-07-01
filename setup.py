#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup, find_packages
import versioneer

pkg_name = 'ffsubsync'

__version__ = versioneer.get_version()
if 'unknown' in __version__.lower():
    with open(os.path.join(pkg_name, '__version__')) as f:
        __version__ = f.read().strip()


def read_file(fname):
    with open(fname, 'r') as f:
        return f.read()


history = read_file('HISTORY.rst')
requirements = read_file('requirements.txt').strip().split()
setup(
    name=pkg_name,
    version=__version__,
    cmdclass=versioneer.get_cmdclass(),
    author='Stephen Macke',
    author_email='stephen.macke@gmail.com',
    description='Language-agnostic synchronization of subtitles with video.',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/smacke/ffsubsync',
    packages=find_packages(exclude=['docs']),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'ffs = ffsubsync:main',
            'subsync = ffsubsync:main',
            'ffsubsync = ffsubsync:main'
        ],
    },
    license='MIT',
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Multimedia :: Sound/Audio :: Speech',
        'Topic :: System :: Systems Administration',
        'Topic :: Utilities',
    ],
)

# python setup.py sdist
# twine upload dist/*
