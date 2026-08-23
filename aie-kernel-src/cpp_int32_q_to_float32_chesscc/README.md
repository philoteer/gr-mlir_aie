# int32 Q format to float32

Build this example with `make FRACTIONAL_BITS=16` (or append `NPU2=1` for
NPU2). It produces `build/final.xclbin` and `build/insts.bin`.

`FRACTIONAL_BITS` configures the format; its valid range is 0 through 31 and
the integer-bit count is `31 - FRACTIONAL_BITS`. The kernel converts signed
`int32` fixed-point samples to float32 by scaling by `2^-FRACTIONAL_BITS`. Its
input is compatible with `float32_to_int32_Q` using the same format.
