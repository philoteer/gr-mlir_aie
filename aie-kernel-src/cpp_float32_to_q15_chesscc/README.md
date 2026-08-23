# float32 to Q0.15

Build this example with `make` (or `make NPU2=1` for NPU2). It produces
`build/final.xclbin` and `build/insts.bin`.

The kernel assumes in-range float input and converts it to signed Q0.15
`int16` samples by scaling by `32768` and truncating toward zero. Its output
is compatible with the `short` stream accepted by `Q15_to_float32`.
