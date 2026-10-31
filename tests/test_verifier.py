from __future__ import annotations

import json
from pathlib import Path

import pytest
from slimconfig import build_output, plain_output

import verifier
from contract import TASKS, TOOLS, CountInput, Dataset, ExactOutput, FindInput, SketchInput, Task, Tool
from plugin import build_sketch, count_pairs, exact_frequent_pairs
from plugin.bundle import LEDGER

EDGE = {"dataset": "edge-cases", "min_count": 3, "memory_budget": 16, "query_budget": 8, "min_recall": "1.0"}
EMPTY = {"dataset": "acl-small", "min_count": 100, "memory_budget": 500, "query_budget": 100, "min_recall": "1.0"}
FULL = {"dataset": "acl-full", "min_count": 55, "memory_budget": 3000, "query_budget": 400, "min_recall": "0.95"}
AT_K = {"a": "alpha", "b": "beta"}  # count exactly k = 3


# The verifier gets contract instances, as NanoAgent's sandbox builds them from JSON.
def verify_task(arguments: dict, pairs: list[dict], workspace: Path, task: dict | None = None) -> dict:
    """The verdict on ``pairs`` for the task ``arguments``, whose reference exact_frequent_pairs computes from
    ``task`` (the same arguments unless given)."""
    given, made = TASKS[Task.FIND_FREQUENT_PAIRS]
    reference = plain_output(exact_frequent_pairs(build_output(task or arguments, given), workspace / "scratch"))
    return verifier.verify_task(
        Task.FIND_FREQUENT_PAIRS.value,
        build_output(arguments, given),
        build_output({"pairs": pairs}, made),
        build_output(reference, ExactOutput),
        str(workspace),
    )


def verify_input(tool: Tool, value: dict) -> dict:
    return verifier.verify_input(tool.value, build_output(value, TOOLS[tool][0]))


def reference(arguments: dict) -> list[dict]:
    found = exact_frequent_pairs(build_output(arguments, FindInput), Path("."))
    return [{"a": pair.a, "b": pair.b} for pair in found.pairs]


def solved(workspace: Path, arguments: dict, width: int, depth: int, last_rank: int | None = None) -> list[dict]:
    """Build a sketch, count ranks 1 to ``last_rank`` (every candidate when None) and return the qualifying pairs."""
    dataset = Dataset(arguments["dataset"])
    k, memory, query = arguments["min_count"], arguments["memory_budget"], arguments["query_budget"]
    made = build_sketch(SketchInput(dataset, width, depth, k, memory), workspace)
    found = count_pairs(
        CountInput(dataset, made.sketch_id, 1, made.candidates if last_rank is None else last_rank, k, query), workspace
    )
    return [{"a": pair.a, "b": pair.b} for pair in found.qualifying]


@pytest.fixture
def edge(tmp_path: Path) -> list[dict]:
    """Every candidate of an 8 x 2 edge-cases sketch counted in ``tmp_path``; the qualifying pairs."""
    found = solved(tmp_path, EDGE, 8, 2)
    assert sorted(found, key=lambda pair: (pair["a"], pair["b"])) == reference(EDGE)
    return found


def test_approves_every_qualifying_pair(tmp_path: Path, edge: list[dict]) -> None:
    verdict = verify_task(EDGE, edge, tmp_path)
    assert verdict["valid"] is True and verdict["score"] == 1.0
    assert verdict["details"] == {"submitted": 2, "reference": 2, "counters": 16, "counted": 3}


def test_the_verdict_is_the_same_twice(tmp_path: Path, edge: list[dict]) -> None:
    assert verify_task(EDGE, edge, tmp_path) == verify_task(EDGE, edge, tmp_path)


def test_a_pair_at_exactly_k_qualifies_and_one_at_k_minus_1_does_not(tmp_path: Path, edge: list[dict]) -> None:
    assert AT_K in edge
    assert verify_task(EDGE | {"min_recall": "0.5"}, [AT_K], tmp_path)["valid"] is True
    verdict = verify_task(EDGE, [*edge, {"a": "alpha", "b": "gamma"}], tmp_path)
    assert verdict["valid"] is False and verdict["score"] == 0.0 and "do not qualify" in verdict["message"]


def test_a_concept_repeated_in_a_paper_counts_once(tmp_path: Path, edge: list[dict]) -> None:
    ledger = json.loads((tmp_path / LEDGER).read_text())
    counts = {(pair["a"], pair["b"]): pair["count"] for pair in ledger["counts"][0]["qualifying"]}
    assert counts[("beta", "gamma")] == 4  # 4 papers, one of them listing gamma twice
    assert verify_task(EDGE, edge, tmp_path)["valid"] is True


def test_refuses_a_collided_pair_the_sketch_overestimates(tmp_path: Path, edge: list[dict]) -> None:
    verdict = verify_task(EDGE, [*edge, {"a": "delta", "b": "iota"}], tmp_path)
    assert verdict["valid"] is False and verdict["score"] == 0.0 and "do not qualify" in verdict["message"]


def test_refuses_an_empty_answer_when_pairs_qualify(tmp_path: Path, edge: list[dict]) -> None:
    verdict = verify_task(EDGE, [], tmp_path)
    assert verdict["valid"] is False and verdict["score"] == 0.0 and "recall 0.000" in verdict["message"]


def test_approves_an_empty_answer_when_no_pair_qualifies(tmp_path: Path) -> None:
    assert reference(EMPTY) == []
    assert verify_task(EMPTY, [], tmp_path)["valid"] is True  # no ledger: nothing was counted
    assert solved(tmp_path, EMPTY, 250, 2, last_rank=None) == []
    verdict = verify_task(EMPTY, [], tmp_path)
    assert verdict["valid"] is True and verdict["score"] == 1.0 and verdict["details"]["counters"] == 500


def test_accepts_a_reversed_pair_but_not_a_repeated_one(tmp_path: Path, edge: list[dict]) -> None:
    reversed_pairs = [{"a": pair["b"], "b": pair["a"]} for pair in edge]
    assert verify_task(EDGE, reversed_pairs, tmp_path)["valid"] is True
    for repeat in (edge[0], reversed_pairs[0]):
        verdict = verify_task(EDGE, [*edge, repeat], tmp_path)
        assert verdict["valid"] is False and "submitted twice" in verdict["message"]


@pytest.mark.parametrize(
    ("pair", "message"),
    [
        ({"a": "alpha", "b": "no such concept"}, "is not a concept of edge-cases"),
        ({"a": "alpha", "b": "alpha"}, "with itself"),
    ],
)
def test_refuses_a_malformed_pair(tmp_path: Path, edge: list[dict], pair: dict, message: str) -> None:
    verdict = verify_task(EDGE, [*edge, pair], tmp_path)
    assert verdict["valid"] is False and message in verdict["message"]


def test_refuses_sketches_over_the_memory_budget(tmp_path: Path, edge: list[dict]) -> None:
    ledger = json.loads((tmp_path / LEDGER).read_text())
    (tmp_path / LEDGER).write_text(json.dumps(ledger | {"memory_budget": 10}))
    verdict = verify_task(EDGE | {"memory_budget": 10}, edge, tmp_path, task=EDGE)
    assert verdict["valid"] is False and "use 16 counters, over the memory budget of 10" in verdict["message"]


def test_refuses_a_rebuilt_sketch_charged_nothing_the_first_time(tmp_path: Path, edge: list[dict]) -> None:
    ledger = json.loads((tmp_path / LEDGER).read_text())
    ledger["sketches"][0]["charged"] = 0
    (tmp_path / LEDGER).write_text(json.dumps(ledger))
    verdict = verify_task(EDGE, edge, tmp_path)
    assert verdict["valid"] is False and "charges sketch edge-cases-w8-d2 0 counters, not 16" in verdict["message"]


def test_refuses_counts_over_the_query_budget(tmp_path: Path, edge: list[dict]) -> None:
    ledger = json.loads((tmp_path / LEDGER).read_text())
    (tmp_path / LEDGER).write_text(json.dumps(ledger | {"query_budget": 2}))
    verdict = verify_task(EDGE | {"query_budget": 2}, edge, tmp_path, task=EDGE)
    assert verdict["valid"] is False and "3 distinct pairs were counted, over the query budget of 2" in verdict["message"]


@pytest.mark.parametrize(
    "changes",
    [{"memory_budget": 20}, {"query_budget": 9}, {"min_count": 2}, {"dataset": "acl-small"}],
    ids=["memory_budget", "query_budget", "min_count", "dataset"],
)
def test_refuses_a_ledger_of_another_task(tmp_path: Path, edge: list[dict], changes: dict) -> None:
    verdict = verify_task(EDGE | changes, [], tmp_path)
    assert verdict["valid"] is False and "the ledger is for another" in verdict["message"]


def test_refuses_a_ledger_that_counts_a_pair_twice(tmp_path: Path, edge: list[dict]) -> None:
    ledger = json.loads((tmp_path / LEDGER).read_text())
    ledger["counts"].append(ledger["counts"][0] | {"qualifying": []})
    (tmp_path / LEDGER).write_text(json.dumps(ledger))
    verdict = verify_task(EDGE, edge, tmp_path)
    assert verdict["valid"] is False and "counts a pair twice" in verdict["message"]


def test_refuses_pairs_without_a_ledger(tmp_path: Path, edge: list[dict]) -> None:
    (tmp_path / LEDGER).unlink()
    verdict = verify_task(EDGE, edge, tmp_path)
    assert verdict["valid"] is False and f"no {LEDGER}" in verdict["message"]


def test_refuses_a_qualifying_pair_that_was_never_counted(tmp_path: Path) -> None:
    found = solved(tmp_path, EDGE, 8, 2, last_rank=1)
    pairs = reference(EDGE)
    assert len(found) == 1 and len(pairs) == 2
    verdict = verify_task(EDGE, pairs, tmp_path)
    assert verdict["valid"] is False and "never returned as qualifying" in verdict["message"]
    assert verify_task(EDGE | {"min_recall": "0.5"}, found, tmp_path)["valid"] is True


def test_refuses_a_pair_recorded_with_another_count(tmp_path: Path, edge: list[dict]) -> None:
    ledger = json.loads((tmp_path / LEDGER).read_text())
    ledger["counts"][0]["qualifying"][0]["count"] += 1
    (tmp_path / LEDGER).write_text(json.dumps(ledger))
    verdict = verify_task(EDGE, edge, tmp_path)
    assert verdict["valid"] is False and "not its exact" in verdict["message"]


def test_recall_exactly_at_min_recall_passes(tmp_path: Path, edge: list[dict]) -> None:
    half = EDGE | {"min_recall": "0.5"}
    verdict = verify_task(half, edge[:1], tmp_path)
    assert verdict["valid"] is True and verdict["score"] == 0.5
    verdict = verify_task(EDGE | {"min_recall": "0.501"}, edge[:1], tmp_path)
    assert verdict["valid"] is False and verdict["score"] == 0.5


def test_the_full_instance_passes_a_good_sketch_and_fails_a_shallow_one(tmp_path: Path) -> None:
    good = solved(tmp_path / "good", FULL, 1500, 2, last_rank=400)
    verdict = verify_task(FULL, good, tmp_path / "good")
    assert verdict["valid"] is True and verdict["details"]["submitted"] == 142
    shallow = solved(tmp_path / "shallow", FULL, 3000, 1, last_rank=400)
    verdict = verify_task(FULL, shallow, tmp_path / "shallow")
    assert verdict["valid"] is False and "recall 0.616 is below 0.95" in verdict["message"]


@pytest.mark.parametrize(
    ("tool", "value", "valid"),
    [
        (Tool.EXACT_FREQUENT_PAIRS, EDGE, True),
        (Tool.EXACT_FREQUENT_PAIRS, EDGE | {"min_recall": "1"}, True),
        (Tool.EXACT_FREQUENT_PAIRS, EDGE | {"min_recall": "0.9e0"}, False),
        (Tool.EXACT_FREQUENT_PAIRS, EDGE | {"min_recall": "0"}, False),
        (Tool.EXACT_FREQUENT_PAIRS, EDGE | {"min_count": 0}, False),
        (Tool.EXACT_FREQUENT_PAIRS, EDGE | {"query_budget": 0}, False),
        (Tool.BUILD_SKETCH, {"dataset": "acl-full", "width": 1500, "depth": 2, "min_count": 55, "memory_budget": 3000}, True),
        (Tool.BUILD_SKETCH, {"dataset": "acl-full", "width": 1501, "depth": 2, "min_count": 55, "memory_budget": 3000}, False),
        (Tool.BUILD_SKETCH, {"dataset": "acl-full", "width": 10, "depth": 17, "min_count": 55, "memory_budget": 3000}, False),
        (Tool.BUILD_SKETCH, {"dataset": "acl-full", "width": 0, "depth": 2, "min_count": 55, "memory_budget": 3000}, False),
    ],
)
def test_checks_what_the_types_cannot_say(tool: Tool, value: dict, valid: bool) -> None:
    assert verify_input(tool, value)["valid"] is valid


@pytest.mark.parametrize(
    ("changes", "valid"),
    [({}, True), ({"sketch_id": "acl-full-w8-d2"}, False), ({"sketch_id": "acl-small-8x2"}, False), ({"count": 0}, False)],
)
def test_checks_a_count_call(changes: dict, valid: bool) -> None:
    value = {"dataset": "acl-small", "sketch_id": "acl-small-w333-d3", "start_rank": 1, "count": 100}
    assert verify_input(Tool.COUNT_PAIRS, value | {"min_count": 10, "query_budget": 500} | changes)["valid"] is valid
