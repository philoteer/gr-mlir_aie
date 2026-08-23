# Multi-kernel chain: three compute cores in series (cbfloat16)

This example wires **three AIE compute cores in series**, each running its own
kernel, and demonstrates the throughput loss ("bubbles") caused by not
effectively feeding the pipeline:

```
ShimDMA -> [core0: scale by 2+j] -> [core1: rotate 90deg] -> [core2: add 0.5+j0.25] -> ShimDMA
```

Each stage is a **stateful kernel** - it keeps an internal phase counter (a
small state machine) that advances once per processed tile and modulates the
computation, so the kernels are **not memoryless**. None of the stages is a
passthrough, so a mis-wired pipeline or a broken kernel state shows up
immediately as wrong output.

## Stateful kernels (for debugging host code)

Each stage lives in its own file and carries its own internal state:

| file                | state machine                                  | operation per tile           |
|---------------------|------------------------------------------------|------------------------------|
| `stage_scale_mul.cc`| 4-state phase counter (mod 4)                  | `y = (2+j1) * j^(g%4) * x`   |
| `stage_rotate90.cc` | 3-state phase counter (mod 3)                  | `y = j^((g%3)+1) * x`        |
| `stage_add_const.cc`| 2-state phase counter (mod 2)                  | `y = x + (0.5+j0.25)*(-1)^(g%2)` |

`g` is the **global tile index**: how many tiles have been fed to the core
since it was last (re)loaded.  The cores run forever (`while_true` in
`aie2.py`), so the internal phase persists across host invocations.  For
global tile `g` the chain computes:

```
y = ((x * (2+j1) * j^(g%4)) * j^((g%3)+1)) + (0.5+j0.25) * (-1)^(g%2)
```

### Why state matters for host-code debugging

A **memoryless** kernel produces identical output for identical input no
matter what the host does around it.  A **stateful** kernel only produces the
expected output if the host feeds tiles continuously and in order from the
moment the kernels were loaded:

* a correct host tracks the global tile index `g` and its reference matches
  (`python3 test.py` PASSes);
* a host bug that resets/reloads the NPU kernels mid-stream - while the
  kernels are still in execution - restarts every internal phase at 0.  If
  that host keeps using its own continuous `g`, the outputs no longer match,
  and the mismatch is large and unmistakable.

Run the demonstration with:

```sh
python3 test.py --break-state
```

which shows the outputs still matching the "fresh" state (`g=0`) while
differing wildly from the host's continuous `g`.

## The bubble knob: inter-core ObjectFifo depth

The inter-stage object fifos are single-buffered (`depth=1`) by default.
While core N is computing on a buffer it also blocks core N-1 from filling
the next buffer, so all three stages serialize per tile and idle gaps
(bubbles) propagate down the chain:

- `FIFO_DEPTH=1` : every tile pays `t_stage0 + t_stage1 + t_stage2`
  (serialized handoff).
- `FIFO_DEPTH=2` (or more): double buffering lets core N-1 fill the next
  buffer while core N still computes on the current one; steady-state
  throughput becomes limited by the *slowest* stage only.

Build and compare on hardware:

```sh
make FIFO_DEPTH=1 && make test   # bubbled pipeline
make FIFO_DEPTH=2 && make test   # properly fed pipeline
```

`make test` prints the median run time over repeated invocations; the depth=2
build should be noticeably faster end-to-end. For a bigger effect, replace the
stage kernels with heavier per-tile math so that compute dominates DMA time.

## Files

| file                | purpose                                              |
|---------------------|------------------------------------------------------|
| `stage_scale_mul.cc`| stage 1 kernel: stateful scale by `(2+j)*j^(g%4)`    |
| `stage_rotate90.cc` | stage 2 kernel: stateful rotate by `j^((g%3)+1)`     |
| `stage_add_const.cc`| stage 3 kernel: stateful add `(0.5+j0.25)*(-1)^(g%2)`|
| `aie2.py`           | iron design: 3 workers chained via ObjectFifos       |
| `test.py`           | host-side correctness check + state-break demo + timing|
| `Makefile`          | chesscc-based build (`npu` default, `NPU2=1` for npu2)|

## GNU Radio usage

Build with `make`, then use the generic `mlir_aie_cpp_bfloat16` GRC block with
`path_xclbin = .../cpp_cbfloat16_multikernel_chain_chesscc/build/final.xclbin`,
`path_insts_bin = .../build/insts.bin`, kernel name `MLIR_AIE` and
`VECTOR_SIZE = 8192` (4096 complex samples x 2 components). Wrap it with the
`complex64_to_cbfloat` / `bfloat16_to_float32` converter blocks like the other
cbfloat16 examples in this repository.

Because the kernels are stateful, the GNU Radio block (and any host code)
must keep feeding the pipeline continuously without resetting/reloading the
kernel; otherwise the state machine phases restart and the output visibly
diverges from the expected continuous behavior.