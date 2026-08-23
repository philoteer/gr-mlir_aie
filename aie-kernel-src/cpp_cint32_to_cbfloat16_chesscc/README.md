# cint32 to cbfloat16

Build this example with `make FRACTIONAL_BITS=16` (or append `NPU2=1` for
NPU2). It produces `build/final.xclbin` and `build/insts.bin`.

`FRACTIONAL_BITS` configures the Q format; its valid range is 0 through 31 and
the integer-bit count is `31 - FRACTIONAL_BITS`. The kernel converts each
packed `cint32` component to `cbfloat16` by scaling by `2^-FRACTIONAL_BITS`.
At the IRON boundary both types are interleaved real and imaginary components,
so its input is compatible with `complex64_to_cint32` using the same format.
