# Q0.15 to bfloat16

Build this example with `make` (or `make NPU2=1` for NPU2). It produces
`build/final.xclbin` and `build/insts.bin`.

The kernel accepts signed Q0.15 `int16` samples and converts them to
`bfloat16` by multiplying by `1/32768`. Its input is directly compatible with
the `short` stream emitted by `float32_to_Q15`.
