"""count_pairs: count a rank range of a sketch's candidates exactly, within the task's query budget.

Every call is recorded in ledger.json, which build_sketch started. The first call fixes the query_budget;
later calls must pass the same. A pair any earlier call counted is skipped, neither counted nor charged again,
so the query budget counts distinct pairs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from contract import CountedPair, CountInput, CountOutput
from plugin.bundle import candidates_path, load, read_ledger, sketch_id_pattern, write_ledger


def _ledger(given: CountInput, workspace: Path) -> dict:
    """The ledger so far, its query_budget fixed by this call when it is the first count_pairs call."""
    ledger = read_ledger(workspace)
    if ledger is None:
        raise ValueError("no sketch is built yet: call build_sketch first, then count its candidates")
    for field, value in (("dataset", given.dataset.value), ("min_count", given.min_count)):
        if ledger[field] != value:
            raise ValueError(f"this task's ledger is for {field} {ledger[field]}: pass {field} {ledger[field]}")
    if ledger["query_budget"] is None:
        ledger["query_budget"] = given.query_budget
    elif ledger["query_budget"] != given.query_budget:
        raise ValueError(
            f"this task's ledger is for query_budget {ledger['query_budget']}: pass query_budget {ledger['query_budget']}"
        )
    return ledger


def count_pairs(given: CountInput, workspace: Path) -> CountOutput:
    if given.min_count < 1:
        raise ValueError(f"min_count must be at least 1, got {given.min_count}: copy it from the task input")
    if given.query_budget < 1:
        raise ValueError(f"query_budget must be at least 1, got {given.query_budget}: copy it from the task input")
    if given.start_rank < 1 or given.count < 1:
        raise ValueError(f"start_rank and count must each be at least 1, got {given.start_rank} and {given.count}")
    if not re.fullmatch(sketch_id_pattern(given.dataset), given.sketch_id):
        raise ValueError(
            f"sketch_id {given.sketch_id!r} is not a {given.dataset.value} sketch:"
            f" pass a sketch_id build_sketch returned for it, {given.dataset.value}-w<width>-d<depth>"
        )
    ledger = _ledger(given, workspace)
    path = candidates_path(workspace, given.sketch_id)
    if not path.exists() or all(call["sketch_id"] != given.sketch_id for call in ledger["sketches"]):
        raise ValueError(f"sketch {given.sketch_id} is not built: call build_sketch for it first")
    candidates = json.loads(path.read_text())["candidates"]
    first, last = given.start_rank, given.start_rank + given.count - 1
    if first > len(candidates):
        raise ValueError(f"sketch {given.sketch_id} has {len(candidates)} candidates: start_rank must be at most that")
    if last > len(candidates):
        raise ValueError(
            f"ranks {first} to {last} go past sketch {given.sketch_id}'s {len(candidates)} candidates:"
            f" with start_rank {first}, count may be at most {len(candidates) - first + 1}"
        )

    seen = {(a, b) for call in ledger["counts"] for a, b in call["counted"]}
    remaining = given.query_budget - len(seen)
    span = [(a, b) for a, b, _ in candidates[first - 1 : last]]
    new = [pair for pair in span if pair not in seen]
    if len(new) > remaining:
        fits, fresh = 0, 0
        for pair in span:
            fresh += pair not in seen
            if fresh > remaining:
                break
            fits += 1
        raise ValueError(
            f"ranks {first} to {last} hold {len(new)} pairs not yet counted ({len(span) - len(new)} already counted),"
            f" but only {remaining} of the query budget {given.query_budget} remain: with start_rank {first},"
            f" count may be at most {fits}"
            + ("; the query budget is used up, so submit what you found" if not remaining else "")
        )

    corpus = load(given.dataset)
    index = {concept: i for i, concept in enumerate(corpus.concepts)}
    qualifying = []
    for a, b in new:
        i, j = sorted((index[a], index[b]))
        count = corpus.pair_counts.get((i, j), 0)
        if count >= given.min_count:
            qualifying.append(CountedPair(a=a, b=b, count=count))
    ledger["counts"].append(
        {
            "sketch_id": given.sketch_id,
            "start_rank": first,
            "count": given.count,
            "counted": [list(pair) for pair in new],
            "skipped": len(span) - len(new),
            "qualifying": [{"a": pair.a, "b": pair.b, "count": pair.count} for pair in qualifying],
        }
    )
    write_ledger(workspace, ledger)
    total = len(seen) + len(new)
    return CountOutput(
        qualifying=qualifying,
        counted=len(new),
        skipped=len(span) - len(new),
        total_counted=total,
        remaining=given.query_budget - total,
    )
