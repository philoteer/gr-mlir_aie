#
# Copyright 2008,2009 Free Software Foundation, Inc.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

# The presence of this file turns this directory into a Python package

'''
This is the GNU Radio MLIR_AIE module. Place your Python package
description here (python/__init__.py).
'''
import os

# import pybind11 generated symbols into the mlir_aie namespace
try:
    # this might fail if the module is python-only
    from .mlir_aie_python import *
except ModuleNotFoundError:
    pass

# import any pure python here
from .mlir_aie_python_uint8 import mlir_aie_python_uint8
from .mlir_aie_python_int32 import mlir_aie_python_int32
from .mlir_aie_python_bfloat16 import mlir_aie_python_bfloat16
from .mlir_aie_python_tagged_int32_to_int64 import mlir_aie_python_tagged_int32_to_int64
from .mlir_aie_python_tagged_int32_to_int32 import mlir_aie_python_tagged_int32_to_int32
#
