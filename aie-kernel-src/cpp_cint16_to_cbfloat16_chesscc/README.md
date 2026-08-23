# cint16 to cbfloat16

Build this example with `make` (or `make NPU2=1` for NPU2). It produces
`build/final.xclbin` and `build/insts.bin`.

The kernel accepts `cint16` samples in Q1.15 format and produces `cbfloat16`
samples. At the IRON boundary both types are represented as interleaved real
and imaginary 16-bit components, so each input and output complex sample is
four bytes. This is compatible with the packed `int` stream emitted by
`complex64_to_cint16`.
