# gr-mlir-aie
GNU Radio with MLIR-AIE. May contain GPL codes from GNU Radio (therefore licensed in GPL 3.0). MLIR codes from AMD are licensed under Apache 2.0 (which is GPL-compatible).

## SigMF frequency and frame boundary tags

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
tag at their exact `core:sample_start`. When `core:sample_count` is present,
they also emit `frame_end` at `core:sample_start + core:sample_count - 1`, the
last sample of the annotated frame. Counts must be positive integers and the
end offset must fit uint64. Annotations without counts retain start-only
tagging. Both tag values are PMT dictionaries:

| Field | PMT type | Meaning |
| --- | --- | --- |
| `frame_id` | symbol | Annotation label, e.g. `wifi-frame-00000000`; unique within the metadata |
| `sample_start` | uint64 | Original frame-start offset in the recording (both events) |
| `sample_offset` | uint64 | Original offset of this event: frame start or last frame sample |
| `timestamp_ns` | uint64 | `CLOCK_MONOTONIC` nanoseconds at callback entry |

For the generator's `cf32_le` data, use a Complex Float32 File Source and the
injector's default `itemsize=8`. Frame tagging is automatic; frequency tagging
continues to use `rx_freq` or the configured frequency key. Unrelated
annotations are ignored. Frame labels must be unique; annotation order does
not matter. An initial seek skips earlier events independently: when seeking
into a frame, its start is skipped but its end is still tagged.

At the decoded-PDU-ready endpoint, use `frame_id` to match the PDU to the
appropriate event, then calculate latency using `CLOCK_MONOTONIC` explicitly:

```python
import time
import pmt

timestamp_ns = pmt.to_uint64(pmt.dict_ref(
    tag.value, pmt.intern("timestamp_ns"), pmt.PMT_NIL))
latency_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC) - timestamp_ns
```

Use `frame_start` for start-to-decode latency or `frame_end` for
last-frame-sample-ingress-to-decode latency. C++ endpoints should likewise use
`clock_gettime(CLOCK_MONOTONIC, ...)`.

One timestamp is taken at the beginning of each `work()` callback, before
sample copying and frequency/frame tag processing. All start/end events in
that buffer share the timestamp, including single-sample frames. This measures
buffer ingress into the injector, not individual sample arrival. Scheduler
batching and downstream buffering contribute to measured latency; there is no
sample-rate interpolation. IDs are recording-local, so include a run/recording
identifier when aggregating separate experiments.

### Named intermediate measurement points

`mlir_aie.latency_tag_updater(name, itemsize=8)` passes a stream through and
appends a named timestamp to each `frame_start`/`frame_end` tag from the injector.
It is available in GRC as **Latency Tag Updater**, with Measurement Name, Sample
Type, and Vector Length parameters. In Python, `itemsize` is the complete item
size in bytes, including vector length.

```python
first = mlir_aie.latency_tag_updater("after_sync")
second = mlir_aie.latency_tag_updater("before_decode")
# Insert first and second at the desired stream measurement points.
```

The original `timestamp_ns`, frame metadata, tag offsets, and source IDs remain
intact. Each updater extends a nested PMT dictionary named `timestamps_ns`:

```text
frame_end = {
    frame_id: wifi-frame-00000000,
    sample_start: 100000,
    sample_offset: 108080,
    timestamp_ns: <injector ingress time>,
    timestamps_ns: {
        after_sync: <first updater ingress time>,
        before_decode: <second updater ingress time>
    }
}
```

All times are uint64 `CLOCK_MONOTONIC` nanoseconds. Each updater takes one
timestamp at callback entry and shares it across events in that buffer. Names
must be nonempty; use a distinct name for each point. If the configured name
already exists, its original entry is retained. Unrelated tags and malformed
timestamp tags pass through unchanged. Each input tag produces exactly one
output tag.

To calculate latency from the injector or between two measurement points:

```python
times = pmt.dict_ref(tag.value, pmt.intern("timestamps_ns"), pmt.PMT_NIL)
after_sync_ns = pmt.to_uint64(pmt.dict_ref(
    times, pmt.intern("after_sync"), pmt.PMT_NIL))
before_decode_ns = pmt.to_uint64(pmt.dict_ref(
    times, pmt.intern("before_decode"), pmt.PMT_NIL))
ingress_ns = pmt.to_uint64(pmt.dict_ref(
    tag.value, pmt.intern("timestamp_ns"), pmt.PMT_NIL))
latency_from_injector_ns = before_decode_ns - ingress_ns
latency_between_points_ns = before_decode_ns - after_sync_ns
```

Run the latency block QA, including the generated recording integration tests, with:

```sh
SIGMF_LATENCY_METADATA=/home/piloteer/xdna_phy/gr-ieee802-11-aie/Experiments/latency/wifi_latency.sigmf-meta \
  ctest --test-dir gr-mlir_aie/build -R 'qa_sigmf_tag_injector|qa_latency_tag_updater' --output-on-failure
```

`mlir_aie_80211_phy` and `mlir_aie_80211_phy_evm` automatically use input
`rx_freq` tags (the injector's default key) to update their nominal frequency.
Tag values are PMT doubles in Hz; the kernel frequency is rounded to integer
MHz. A tag at a `VECTOR_SIZE` boundary applies before that vector runs. A tag
inside a vector applies at the next vector boundary, after all outstanding
runs finish. If several changes occur inside one vector, the last value wins.
Each applied integer-MHz change prints the block name and old/new frequencies.
