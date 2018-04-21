#!/bin/env python
# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

from __future__ import print_function

import glob
import os
import os.path
import sys
import time
import traceback
import urllib
from distutils.core import setup  # , Extension, Command

assert sys.version_info[0] == 2 and sys.version_info[1] >= 7,\
    "requires Python version 2.7 or later, but not Python 3.x"

remote = "https://storage.googleapis.com/tmb-models/"
models = ["lowskew-000000259-011440.pt"]
for model in models:
    if not os.path.exists(model):
        urllib.URLopener().retrieve(remote+model, model)


scripts = """
ocroseg-train
ocroseg-pred
ocroseg-predlines
""".split()

setup(
    name='ocroseg',
    version='v0.0',
    author="Thomas Breuel",
    description="Text line recognition.",
    packages=["ocroseg"],
    scripts=scripts,
    data_files = [("share/ocroseg", models)]
)
