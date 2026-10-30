# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import runpy
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch


COMPAT = Path(__file__).resolve().parents[1] / "iron_compat.py"


class IronCompatibilityTest(unittest.TestCase):
    def load_compat(self, runtime, program, placer=None):
        aie = ModuleType("aie")
        aie.__path__ = []
        iron = ModuleType("aie.iron")
        iron.__path__ = []
        iron.Runtime = runtime
        iron.Program = program
        modules = {"aie": aie, "aie.iron": iron}
        if placer is not None:
            placers = ModuleType("aie.iron.placers")
            placers.SequentialPlacer = placer
            modules["aie.iron.placers"] = placers
        else:
            modules["aie.iron.placers"] = None
        with patch.dict(sys.modules, modules):
            return runpy.run_path(str(COMPAT))

    def test_legacy_api_is_used_unchanged(self):
        class LegacyRuntime:
            def sequence(self):
                pass

        program, placer = object(), object()
        compat = self.load_compat(LegacyRuntime, program, placer)
        self.assertIs(compat["Runtime"], LegacyRuntime)
        self.assertIs(compat["Program"], program)
        self.assertIs(compat["SequentialPlacer"], placer)

    def test_callback_preserves_abi_workers_and_transfer_order(self):
        class CallbackRuntime:
            def __init__(self, callback, args):
                self.callback, self.args = callback, args

        program = Mock()
        compat = self.load_compat(CallbackRuntime, program)
        self.assertIsNone(compat["SequentialPlacer"])
        rt = compat["Runtime"]()
        workers = [object(), object(), object()]
        input_fifo, output_fifo = Mock(), Mock()
        with rt.sequence("input_type", "output_type", "unused_type") as (a, b, _):
            rt.start(*workers)
            rt.fill(input_fifo, a)
            rt.drain(output_fifo, b, wait=True)
        compat["Program"]("device", rt)
        args, kwargs = program.call_args
        self.assertEqual(args[0], "device")
        self.assertEqual(kwargs, {"workers": workers})
        native = args[1]
        self.assertEqual(native.args, ["input_type", "output_type", "unused_type",
                                       input_fifo, output_fifo])
        transfers = Mock()
        transfers.attach_mock(input_fifo, "input")
        transfers.attach_mock(output_fifo, "output")
        native.callback("host_input", "host_output", "unused", input_fifo, output_fifo)
        from unittest.mock import call
        self.assertEqual(transfers.mock_calls, [call.input.fill("host_input"),
                                               call.output.drain("host_output", wait=True)])


if __name__ == "__main__":
    unittest.main()
