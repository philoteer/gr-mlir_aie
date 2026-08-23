# cbfloat16 to cint16

Build this example with `make` (or `make NPU2=1` for NPU2). It produces
`build/final.xclbin` and `build/insts.bin`.

The kernel assumes in-range `cbfloat16` input, converts each component to
signed Q1.15 with truncation, and returns packed `cint16` samples. At the IRON
boundary both types are interleaved real and imaginary 16-bit components, so
the output is compatible with the packed `int` stream accepted by
`cint16_to_complex64`.
