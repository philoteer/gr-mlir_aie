# gr-mlir-aie
GNU Radio with MLIR-AIE. May contain GPL codes from GNU Radio (therefore licensed in GPL 3.0). MLIR codes from AMD are licensed under Apache 2.0 (which is GPL-compatible).

## SigMF frequency and frame-start tags

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
Repeating playback and seeking during playback are not supported.
Building it requires Boost.JSON (Boost 1.75 or newer).

### Latency measurement with synthetic Wi-Fi frames

Annotations with `core:label` beginning `wifi-frame-` (as written by
`gr-ieee802-11-aie/Experiments/latency/wifi_tx_sigmf.grc`) emit a `frame_start`
tag at their exact `core:sample_start`. The tag value is a PMT dictionary:

| Field | PMT type | Meaning |
| --- | --- | --- |
| `frame_id` | symbol | Annotation label, e.g. `wifi-frame-00000000`; unique within the metadata |
| `sample_start` | uint64 | Original sample offset in the recording |
| `timestamp_ns` | uint64 | `CLOCK_MONOTONIC` nanoseconds at tag emission |

For the generator's `cf32_le` data, use a Complex Float32 File Source and the
injector's default `itemsize=8`. Frame tagging is automatic; frequency tagging
continues to use `rx_freq` or the configured frequency key. Unrelated
annotations are ignored. Frame labels must be unique; annotation order does
not matter. An initial seek skips earlier frame starts, including frames whose
body overlaps the seek position.

At a downstream measurement point, extract the dictionary and calculate:

```python
import time
import pmt

timestamp_ns = pmt.to_uint64(pmt.dict_ref(
    tag.value, pmt.intern("timestamp_ns"), pmt.PMT_NIL))
latency_ns = time.monotonic_ns() - timestamp_ns
```

The timestamp is taken during `work()`, not when metadata is loaded. It measures
software passage through the injector; scheduler batching and downstream
buffering contribute to measured latency. It is not the recording's wall-clock
timestamp or an interpolated sample time. IDs are recording-local, so include a
run/recording identifier when aggregating separate experiments.

Run the injector QA, including the generated recording integration test, with:

```sh
SIGMF_LATENCY_METADATA=/home/piloteer/xdna_phy/gr-ieee802-11-aie/Experiments/latency/wifi_latency.sigmf-meta \
  ctest --test-dir gr-mlir_aie/build -R qa_sigmf_tag_injector --output-on-failure
```

`mlir_aie_80211_phy` and `mlir_aie_80211_phy_evm` automatically use input
`rx_freq` tags (the injector's default key) to update their nominal frequency.
Tag values are PMT doubles in Hz; the kernel frequency is rounded to integer
MHz. A tag at a `VECTOR_SIZE` boundary applies before that vector runs. A tag
inside a vector applies at the next vector boundary, after all outstanding
runs finish. If several changes occur inside one vector, the last value wins.
Each applied integer-MHz change prints the block name and old/new frequencies.
