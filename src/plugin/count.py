"""count_pairs: count a rank range of a sketch's candidates exactly, within the task's query budget.

Every call is recorded in the instance's ledger, ledgers/<instance>.json, which build_sketch started. A pair any
earlier call counted is skipped, neither counted nor charged again, so the query budget counts distinct pairs.
The skipped pairs that qualified are returned too, with their recorded counts and previously_counted set, so
that a second run of the same instance in the same workspace can still find them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from contract import CountInput, CountOutput, QualifyingPair
from plugin.bundle import candidates_path, instance, load, read_ledger, sketch_id_pattern, write_ledger


def count_pairs(given: CountInput, workspace: Path) -> CountOutput:
    if given.min_count < 1:
        raise ValueError(f"min_count must be at least 1, got {given.min_count}: copy it from the task input")
    if given.memory_budget < 1:
        raise ValueError(f"memory_budget must be at least 1, got {given.memory_budget}: copy it from the task input")
    if given.query_budget < 1:
        raise ValueError(f"query_budget must be at least 1, got {given.query_budget}: copy it from the task input")
    if given.start_rank < 1 or given.count < 1:
        raise ValueError(f"start_rank and count must each be at least 1, got {given.start_rank} and {given.count}")
    if not re.fullmatch(sketch_id_pattern(given.dataset), given.sketch_id):
        raise ValueError(
            f"sketch_id {given.sketch_id!r} is not a {given.dataset.value} sketch:"
            f" pass a sketch_id build_sketch returned for it, {given.dataset.value}-w<width>-d<depth>"
        )
    name = instance(given.dataset, given.min_count, given.memory_budget, given.query_budget)
    ledger = read_ledger(workspace, name)
    if ledger is None:
        raise ValueError(
            f"no sketch is built for dataset {given.dataset.value}, min_count {given.min_count}, memory_budget"
            f" {given.memory_budget} and query_budget {given.query_budget}: call build_sketch first with the same"
            " values, copied from the task input as given"
        )
    path = candidates_path(workspace, name, given.sketch_id)
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
    earlier = {(pair["a"], pair["b"]): pair["count"] for call in ledger["counts"] for pair in call["qualifying"]}
    fresh_pairs, qualifying = set(new), []
    for a, b in span:
        if (a, b) in fresh_pairs:
            i, j = sorted((index[a], index[b]))
            count = corpus.pair_counts.get((i, j), 0)
            if count >= given.min_count:
                qualifying.append(QualifyingPair(a=a, b=b, count=count, previously_counted=False))
        elif (a, b) in earlier:
            qualifying.append(QualifyingPair(a=a, b=b, count=earlier[a, b], previously_counted=True))
    ledger["counts"].append(
        {
            "sketch_id": given.sketch_id,
            "start_rank": first,
            "count": given.count,
            "counted": [list(pair) for pair in new],
            "skipped": len(span) - len(new),
            "qualifying": [
                {"a": pair.a, "b": pair.b, "count": pair.count} for pair in qualifying if not pair.previously_counted
            ],
        }
    )
    write_ledger(workspace, name, ledger)
    total = len(seen) + len(new)
    return CountOutput(
        qualifying=qualifying,
        counted=len(new),
        skipped=len(span) - len(new),
        total_counted=total,
        remaining=given.query_budget - total,
    )
