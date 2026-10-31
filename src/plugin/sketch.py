"""build_sketch: a Count-Min Sketch of a corpus's pair stream, and the stream's pairs it estimates at min_count.

Pass 1 adds every pair occurrence of the stream to one counter in each of the first depth hash rows: for the
concept indices i < j and V concepts, the key x = i * V + j goes to counter ((a * x + b) mod p) mod width of
the row (a, b). Pass 2 estimates each pair of the stream as the least of its depth counters, which is never
below its true count, and keeps those estimated at min_count or more. The candidates go to
candidates/<sketch_id>.json, highest estimate first, ties in (a, b) order, so that they never pass through the
model. Every call is recorded in ledger.json: the first fixes the dataset, min_count and memory_budget, and a
sketch already built is not charged again.
"""

from __future__ import annotations

import json
from pathlib import Path

from contract import EstimateBucket, SketchInput, SketchOutput
from plugin.bundle import candidates_path, hash_rows, load, names, read_ledger, sketch_id, write_ledger
from plugin.rule import histogram


def _ledger(given: SketchInput, workspace: Path) -> dict:
    """The ledger so far, or a new one fixing this call's dataset, min_count and memory_budget."""
    ledger = read_ledger(workspace)
    if ledger is None:
        return {
            "dataset": given.dataset.value,
            "min_count": given.min_count,
            "memory_budget": given.memory_budget,
            "query_budget": None,
            "sketches": [],
            "counts": [],
        }
    for field, value in (("dataset", given.dataset.value), ("min_count", given.min_count)):
        if ledger[field] != value:
            raise ValueError(f"this task's ledger is for {field} {ledger[field]}: pass {field} {ledger[field]}")
    if ledger["memory_budget"] != given.memory_budget:
        raise ValueError(
            f"this task's ledger is for memory_budget {ledger['memory_budget']}:"
            f" pass memory_budget {ledger['memory_budget']}"
        )
    return ledger


def build_sketch(given: SketchInput, workspace: Path) -> SketchOutput:
    prime, rows = hash_rows()
    width, depth = given.width, given.depth
    if width < 1:
        raise ValueError(f"width must be at least 1, got {width}")
    if not 1 <= depth <= len(rows):
        raise ValueError(f"depth must be from 1 to {len(rows)}, the hash rows there are, got {depth}")
    if given.min_count < 1:
        raise ValueError(f"min_count must be at least 1, got {given.min_count}: copy it from the task input")
    if given.memory_budget < 1:
        raise ValueError(f"memory_budget must be at least 1, got {given.memory_budget}: copy it from the task input")

    ledger = _ledger(given, workspace)
    sketch = sketch_id(given.dataset, width, depth)
    used = sum(call["charged"] for call in ledger["sketches"])
    built = any(call["sketch_id"] == sketch for call in ledger["sketches"])
    charged = 0 if built else width * depth
    if used + charged > given.memory_budget:
        left = given.memory_budget - used
        raise ValueError(
            f"a {width} x {depth} sketch needs {width * depth} counters, but only {left} of the memory budget"
            f" {given.memory_budget} remain, since earlier sketches use {used}: "
            + (f"with depth {depth}, width may be at most {left // depth}" if left >= depth else
               f"no sketch of depth {depth} fits; count with the sketches already built")
        )

    corpus = load(given.dataset)
    size = len(corpus.concepts)
    tables = [[0] * width for _ in range(depth)]
    for (i, j), count in corpus.pair_counts.items():
        for table, (a, b) in zip(tables, rows):
            table[((a * (i * size + j) + b) % prime) % width] += count
    ranked = []
    for i, j in corpus.pair_counts:
        estimate = min(table[((a * (i * size + j) + b) % prime) % width] for table, (a, b) in zip(tables, rows))
        if estimate >= given.min_count:
            ranked.append((-estimate, *names(corpus, (i, j))))
    ranked.sort()

    path = candidates_path(workspace, sketch)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = {
        "sketch_id": sketch,
        "dataset": given.dataset.value,
        "width": width,
        "depth": depth,
        "min_count": given.min_count,
        "candidates": [[a, b, -negative] for negative, a, b in ranked],
    }
    path.write_text(json.dumps(saved) + "\n")
    ledger["sketches"].append({"sketch_id": sketch, "width": width, "depth": depth, "charged": charged})
    write_ledger(workspace, ledger)

    buckets = histogram([-negative for negative, _, _ in ranked], given.min_count)
    return SketchOutput(
        sketch_id=sketch,
        counters_charged=charged,
        memory_used=used + charged,
        memory_remaining=given.memory_budget - used - charged,
        candidates=len(ranked),
        histogram=[EstimateBucket(*bucket) for bucket in buckets],
    )
