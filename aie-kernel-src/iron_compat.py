# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compatibility for the linear runtime sequences used by these examples.

Older IRON records transfers in Runtime.sequence(); newer IRON executes a
callback at resolution time and takes workers explicitly in Program.
"""

from contextlib import contextmanager

from aie.iron import Program as IronProgram, Runtime as IronRuntime

try:
    from aie.iron.placers import SequentialPlacer
except ImportError:
    SequentialPlacer = None


if hasattr(IronRuntime, "sequence"):
    Runtime = IronRuntime
    Program = IronProgram
else:
    class Runtime:
        """Record a straight-line fill/drain sequence for the callback API."""

        def __init__(self):
            self.arg_types = []
            self.workers = []
            self.transfers = []

        @contextmanager
        def sequence(self, *arg_types):
            self.arg_types = list(arg_types)
            yield tuple(range(len(arg_types)))

        def start(self, *workers):
            self.workers.extend(workers)

        def fill(self, fifo, source, **kwargs):
            self.transfers.append(("fill", fifo, source, kwargs))

        def drain(self, fifo, dest, **kwargs):
            self.transfers.append(("drain", fifo, dest, kwargs))

    def Program(device, runtime):
        def sequence(*args):
            for i, (verb, _, index, kwargs) in enumerate(runtime.transfers):
                handle = args[len(runtime.arg_types) + i]
                getattr(handle, verb)(args[index], **kwargs)

        # Passing handles as fn_args binds their shim endpoints before resolution.
        handles = [fifo for _, fifo, _, _ in runtime.transfers]
        rt = IronRuntime(sequence, runtime.arg_types + handles)
        return IronProgram(device, rt, workers=runtime.workers)
