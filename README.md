# gr-mlir-aie
GNU Radio with MLIR-AIE. May contain GPL codes from GNU Radio (therefore licensed in GPL 3.0). MLIR codes from AMD are licensed under Apache 2.0 (which is GPL-compatible).

## SigMF frequency tags

`mlir_aie.sigmf_tag_injector(metadata_file, itemsize=8, tag_key="rx_freq",
start_sample=0)` passes samples through unchanged and reads the corresponding
`.sigmf-meta` file once at construction. It emits a PMT double frequency tag in
Hz at the first known `core:frequency` and at each subsequent frequency change,
using the capture's exact `core:sample_start` offset. Existing stream tags are
preserved. Captures without a frequency retain the previous tuning; repeated
frequencies do not generate extra tags. Sample offsets are 64-bit.

For the `ci16_le` recordings in `/mnt/SUB/raw_capture/`, use this GRC chain:

```text
File Source (Int32, repeat disabled, 206.sigmf-data)
  -> SigMF Tag Injector (Int32 (packed ci16), 206.sigmf-meta)
  -> cint16_to_complex64
  -> downstream processing
```

Each packed Int32 contains one complex int16 sample (I and Q). Alternatively,
put the injector after `cint16_to_complex64` and select Complex Float32.
Do not feed individual interleaved Int16 I/Q components as separate items:
SigMF offsets count complete complex samples.

Python example:

```python
from gnuradio import blocks, gr, mlir_aie

tb = gr.top_block()
source = blocks.file_source(gr.sizeof_int, "/mnt/SUB/raw_capture/206.sigmf-data", False)
tags = mlir_aie.sigmf_tag_injector(
    "/mnt/SUB/raw_capture/206.sigmf-meta", gr.sizeof_int)
convert = mlir_aie.cint16_to_complex64()
tb.connect(source, tags, convert)  # Connect convert to your downstream block.
```

For playback starting partway through a file, set `start_sample` to the File
Source's initial offset in complete samples. The active frequency is then
tagged at output offset zero, and later changes are shifted accordingly.
Repeating playback and seeking during playback are not supported. This block
currently handles only capture center frequencies, not timestamps, sample rate
or annotations. Building it requires Boost.JSON (Boost 1.75 or newer).

`mlir_aie_80211_phy` and `mlir_aie_80211_phy_evm` automatically use input
`rx_freq` tags (the injector's default key) to update their nominal frequency.
Tag values are PMT doubles in Hz; the kernel frequency is rounded to integer
MHz. A tag at a `VECTOR_SIZE` boundary applies before that vector runs. A tag
inside a vector applies at the next vector boundary, after all outstanding
runs finish. If several changes occur inside one vector, the last value wins.
Each applied integer-MHz change prints the block name and old/new frequencies.
