# Shifted 64-point `cint16` FFT

This example computes independent forward 64-point FFT frames with the staged
AIE API. It uses three radix-4 stages and applies `fftshift` to each frame so
that negative-frequency bins precede non-negative-frequency bins.

The `fft64_cint16` kernel accepts interleaved complex Q0.15 `cint16` samples and
returns interleaved `cint32` components. The transform is unnormalized: an
input with integer components `x` produces the integer-domain equivalent of
`fft(x)`. The six bits of FFT growth are retained in `cint32`, avoiding the
overflow that a full-scale coherent input would cause in `cint16`.

The `count` argument is the number of complex samples and must be a multiple of
64. Each consecutive group of 64 samples is transformed independently.

## Output format

The output is interleaved complex `cint32` in Q6.15 format. The FFT is
unnormalized, so the six bits of growth from the 64-point transform are kept.
Bins are in `fftshift` order, with DC at output index 32.

`fft64_twiddles.h` is generated deterministically from the staged-FFT formula:

```sh
python3 generate_twiddles.py > fft64_twiddles.h
```

The generator scales each complex component by 32768, truncates toward zero,
and saturates to the signed 16-bit range. It also emits radix-4 tables in the
API-required `T2`, `T1`, `T3` argument order.

Build with `make` and run on hardware with `make test`. Set `NPU2=1` to target
NPU2 instead of NPU1.
