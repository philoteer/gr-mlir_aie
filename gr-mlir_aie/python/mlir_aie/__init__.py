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

# C++ XRT blocks do not need the compiler's Python environment (which can
# use a different NumPy ABI than GNU Radio). Load Python/IRON blocks only
# when requested, so importing the C++ blocks works with system Python.
_python_blocks = {
    'mlir_aie_python_uint8',
    'mlir_aie_python_int32',
    'mlir_aie_python_bfloat16',
    'mlir_aie_python_tagged_int32_to_int64',
    'mlir_aie_python_tagged_int32_to_int32',
}
__all__ = [name for name in globals() if not name.startswith('_')] + sorted(_python_blocks)


def __getattr__(name):
    if name not in _python_blocks:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    from importlib import import_module
    block = getattr(import_module(f'.{name}', __name__), name)
    globals()[name] = block
    return block
