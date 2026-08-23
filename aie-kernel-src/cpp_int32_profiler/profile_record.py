#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np


PAYLOAD_WORDS = 1024
PROFILE_WORDS = 16
PROFILE_MAGIC = 0x50524631
PROFILE_VERSION = 1
FIELDS = (
    "magic",
    "version",
    "total_cycles",
    "vector_cycles",
    "scalar_cycles",
    "checksum_cycles",
    "payload_words",
    "checksum",
    "run_count",
    "reserved_0",
    "reserved_1",
    "reserved_2",
    "reserved_3",
    "reserved_4",
    "reserved_5",
    "reserved_6",
)
CYCLE_FIELDS = (
    "total_cycles",
    "vector_cycles",
    "scalar_cycles",
    "checksum_cycles",
)


def parse_profile(output):
    """Validate and decode the profile record at the end of an output tile."""
    array = np.asarray(output, dtype=np.int32)
    if array.size != PAYLOAD_WORDS + PROFILE_WORDS:
        raise ValueError(f"expected {PAYLOAD_WORDS + PROFILE_WORDS} words, got {array.size}")

    raw = array[-PROFILE_WORDS:]
    values = dict(zip(FIELDS, raw.astype(np.uint32).astype(np.uint64)))
    if values["magic"] != PROFILE_MAGIC:
        raise ValueError(f"bad profile magic 0x{values['magic']:08x}")
    if values["version"] != PROFILE_VERSION:
        raise ValueError(f"unsupported profile version {values['version']}")
    if values["payload_words"] != PAYLOAD_WORDS:
        raise ValueError(f"profile reports {values['payload_words']} payload words")
    return values


def expected_checksum(payload):
    """Return the kernel's wrapping uint32 checksum for payload * 2."""
    total = int(np.asarray(payload, dtype=np.int32).astype(np.uint32).sum(dtype=np.uint64))
    return (total * 2) & 0xFFFFFFFF


def format_report(totals):
    total_cycles = int(totals["total_cycles"])
    runs = int(totals["run_count"])
    lines = [f"profiled runs: {runs}", f"total cycles: {total_cycles}"]
    for field in CYCLE_FIELDS[1:]:
        cycles = int(totals[field])
        percentage = 100.0 * cycles / total_cycles if total_cycles else 0.0
        label = field.removesuffix("_cycles").replace("_", " ")
        lines.append(f"{label:>10}: {cycles:12d} cycles  {percentage:6.2f}%")
    measured = sum(int(totals[field]) for field in CYCLE_FIELDS[1:])
    unaccounted = max(total_cycles - measured, 0)
    percentage = 100.0 * unaccounted / total_cycles if total_cycles else 0.0
    lines.append(f"{'other':>10}: {unaccounted:12d} cycles  {percentage:6.2f}%")
    lines.append(f"average cycles/run: {total_cycles / runs:.1f}")
    return "\n".join(lines)
