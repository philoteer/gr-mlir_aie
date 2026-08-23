# !! Largely vibe-coded, be aware. !!

# AIE cycle-profiler example

This example measures sections of an AIE kernel with
`aie::tile::current().cycles()`. The kernel doubles 1024 `int32` values and
appends a 16-word profile record to the output. The host validates the record
before treating its cycle words as unsigned values and accumulating them in
Python integers.

Build and run on an NPU with:

```sh
make JOBS=1 all
make test
make profile
```

Use `make NPU2=1 all` for an NPU2 target. `profile.py --runs N` controls the
number of samples. Its percentages use the sum of all measured runs, rather
than averaging per-run percentages.

The record layout is defined once in `profile_record.py`: magic, version,
total, vector, scalar, and checksum cycles, payload size, checksum, run count,
and seven reserved words. Counter writes happen after `total_cycles` is read,
so telemetry serialization is intentionally excluded. This output-tail method
is intended for temporary instrumentation; production interfaces should not
silently reserve payload words for profiler data.
