# cbfloat16 to cint32

Build this example with `make FRACTIONAL_BITS=16` (or append `NPU2=1` for
NPU2). It produces `build/final.xclbin` and `build/insts.bin`.

`FRACTIONAL_BITS` configures the Q format; its valid range is 0 through 31 and
the integer-bit count is `31 - FRACTIONAL_BITS`. The kernel assumes input is
in range, scales each `cbfloat16` component by `2^FRACTIONAL_BITS`, truncates,
and writes `cint32`. At the IRON boundary both types are interleaved real and
imaginary components. The output is compatible with the packed `long` stream
accepted by `cint32_to_complex64` configured with the same bit counts.
