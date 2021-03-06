#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is part of datamatrix.

datamatrix is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

datamatrix is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with datamatrix.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
from eyelinkparser import __version__
from setuptools import setup, find_packages

# Increment to force a change in the source tarball.
DUMMY = 1


def get_readme():

    if os.path.exists('readme.md'):
        with open('readme.md') as fd:
            return fd.read()
    return 'No readme information'


setup(
    name=u'python-eyelinkparser',
    version=__version__,
    description=u'An extensible parser for EyeLink data files (EDF)',
    long_description=get_readme(),
    long_description_content_type='text/markdown',
    author=u'Sebastiaan Mathot',
    author_email=u's.mathot@cogsci.nl',
    license=u'GNU GPL Version 3',
    packages=find_packages('.'),
    url=u'https://github.com/smathot/python-eyelinkparser',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
    install_requires=[
        'python-datamatrix',
        'fastnumbers'
        ]
)
