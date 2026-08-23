# float32 to int32 Q format

Build this example with `make FRACTIONAL_BITS=16` (or append `NPU2=1` for
NPU2). It produces `build/final.xclbin` and `build/insts.bin`.

`FRACTIONAL_BITS` configures the format; its valid range is 0 through 31 and
the integer-bit count is `31 - FRACTIONAL_BITS`. The kernel assumes input is
in range, scales by `2^FRACTIONAL_BITS`, truncates, and writes signed `int32`
samples compatible with `int32_Q_to_float32` configured with the same format.
