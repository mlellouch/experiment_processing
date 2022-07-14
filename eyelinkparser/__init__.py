# -*- coding: utf-8 -*-

"""
This file is part of eyelinkparser.

eyelinkparser is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

eyelinkparser is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with datamatrix.  If not, see <http://www.gnu.org/licenses/>.
"""

from datamatrix.py3compat import *
from eyelinkparser._events import sample, fixation, saccade, blink
from eyelinkparser._traceprocessor import defaulttraceprocessor
from eyelinkparser._eyelinkparser import EyeLinkParser
from eyelinkparser._eyelinkplusparser import EyeLinkPlusParser

import tempfile
import os
import shutil

__version__ = u'0.15.0'


def parse(parser=EyeLinkParser, **kwdict):

    return parser(**kwdict).dm


def parse_file(parser=EyeLinkPlusParser, filepath='', **kwdicts):
    assert os.path.isfile(filepath)
    filename = os.path.basename(filepath)
    temp_dir = tempfile.TemporaryDirectory()
    dst = os.path.join(temp_dir.name, filename)
    shutil.copy(src=filepath, dst=dst)

    kwdicts['folder'] = temp_dir.name
    return parser(**kwdicts)


