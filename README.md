# gr-mlir-aie
GNU Radio with MLIR-AIE. May contain GPL codes from GNU Radio (therefore licensed in GPL 3.0). MLIR codes from AMD are licensed under Apache 2.0 (which is GPL-compatible).

* aie-kernel-src: sample XDNA kernel sources; needs Peano and/or CHESS to compile (see https://github.com/xilinx/mlir-aie ).
* gr-mlir-aie: AIE kernel to GNU Radio binding examples. Use the standard GNU Radio OOT module build process (cmake, make, make install, then ldconfig).
* grc-only: An embedded Python-based implementation (not recommended)