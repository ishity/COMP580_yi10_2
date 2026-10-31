from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import pytest
from slimconfig import build_output, plain_output

import verifier
import verifier.rule
from contract import TOOLS, CountInput, Dataset, DescribeInput, FindInput, SketchInput, Tool
from plugin import build_sketch, count_pairs, describe_corpus, exact_frequent_pairs, rule
from plugin.bundle import LEDGER, hash_rows, load

ROOT = Path(__file__).resolve().parents[1]
EDGE = {"dataset": Dataset.EDGE_CASES, "min_count": 3}


def checked(tool: Tool, given, made, workspace: Path) -> dict:
    """verify_output on a tool call, with values built the way the sandbox builds them."""
    given_class, made_class = TOOLS[tool]
    return verifier.verify_output(
        tool.value,
        build_output(plain_output(given), given_class),
        build_output(plain_output(made), made_class),
        str(workspace),
    )


def sketch(workspace: Path, width: int, depth: int, memory_budget: int = 16, **changes):
    given = SketchInput(width=width, depth=depth, memory_budget=memory_budget, **(EDGE | changes))
    return given, build_sketch(given, workspace)


def count(workspace: Path, start: int, number: int, query_budget: int = 8, sketch_id: str = "edge-cases-w8-d2", **changes):
    fields = {"sketch_id": sketch_id, "start_rank": start, "count": number, "query_budget": query_budget}
    given = CountInput(**(EDGE | fields | changes))
    return given, count_pairs(given, workspace)


def slow_counts(dataset: Dataset) -> Counter[tuple[str, str]]:
    """A second exact count, by concept names, straight from the raw JSON."""
    raw = json.loads((ROOT / "src/plugin/data" / f"{dataset.value}.json").read_text())
    names, excluded = raw["keywords"], {frozenset(pair) for pair in raw["excluded_pairs"]}
    counts: Counter[tuple[str, str]] = Counter()
    for paper in raw["papers"]:
        for i, j in combinations(set(paper["concepts"]), 2):
            if frozenset((i, j)) not in excluded:
                counts[tuple(sorted((names[i], names[j])))] += 1
    return counts


def test_describe_corpus_counts_the_bundle(tmp_path: Path) -> None:
    expected = {
        Dataset.ACL_SMALL: (2924, 493, 2597, 7060),
        Dataset.ACL_FULL: (21934, 493, 11009, 57912),
        Dataset.EDGE_CASES: (17, 14, 8, 14),
    }
    for dataset, sizes in expected.items():
        given = DescribeInput(dataset=dataset)
        made = describe_corpus(given, tmp_path)
        assert (made.papers, made.concepts, made.distinct_pairs, made.pair_occurrences) == sizes
        assert made.hash_rows == len(hash_rows()[1]) == verifier.checks.HASH_ROWS
        assert checked(Tool.DESCRIBE_CORPUS, given, made, tmp_path)["valid"] is True


@pytest.mark.parametrize(("width", "depth"), [(1, 1), (37, 1), (100, 3), (500, 2), (1000, 4), (64, 16)])
def test_the_sketch_never_underestimates_a_pair(tmp_path: Path, width: int, depth: int) -> None:
    given = SketchInput(dataset=Dataset.ACL_SMALL, width=width, depth=depth, min_count=1, memory_budget=width * depth)
    build_sketch(given, tmp_path)
    saved = json.loads((tmp_path / "candidates" / f"acl-small-w{width}-d{depth}.json").read_text())
    truth = slow_counts(Dataset.ACL_SMALL)
    estimates = {(a, b): estimate for a, b, estimate in saved["candidates"]}
    assert set(estimates) == set(truth)  # min_count 1: every pair of the stream is a candidate
    assert all(estimates[pair] >= count for pair, count in truth.items())
    if width == 1:
        assert set(estimates.values()) == {sum(truth.values())}


def test_build_sketch_is_deterministic_and_ranked(tmp_path: Path) -> None:
    given = SketchInput(dataset=Dataset.ACL_FULL, width=1500, depth=2, min_count=55, memory_budget=3000)
    first = build_sketch(given, tmp_path / "one")
    second = build_sketch(given, tmp_path / "two")
    assert first == second and first.sketch_id == "acl-full-w1500-d2" and first.candidates == 646
    saved = (tmp_path / "one" / "candidates" / "acl-full-w1500-d2.json").read_text()
    assert saved == (tmp_path / "two" / "candidates" / "acl-full-w1500-d2.json").read_text()
    assert sum(row.candidates for row in first.histogram) == first.candidates
    assert first.histogram[0].first_rank == 1 and first.histogram[-1].last_rank == first.candidates
    assert checked(Tool.BUILD_SKETCH, given, first, tmp_path / "one")["valid"] is True
    run = json.loads(saved)
    run["candidates"].reverse()
    (tmp_path / "one" / "candidates" / "acl-full-w1500-d2.json").write_text(json.dumps(run))
    assert checked(Tool.BUILD_SKETCH, given, first, tmp_path / "one")["valid"] is False


def test_rebuilding_a_sketch_is_not_charged_again(tmp_path: Path) -> None:
    _, made = sketch(tmp_path, 4, 2)
    assert (made.counters_charged, made.memory_used, made.memory_remaining) == (8, 8, 8)
    given, made = sketch(tmp_path, 4, 2)
    assert (made.counters_charged, made.memory_used, made.memory_remaining) == (0, 8, 8)
    assert checked(Tool.BUILD_SKETCH, given, made, tmp_path)["valid"] is True
    given, made = sketch(tmp_path, 8, 1)
    assert (made.counters_charged, made.memory_used, made.memory_remaining) == (8, 16, 0)
    assert checked(Tool.BUILD_SKETCH, given, made, tmp_path)["valid"] is True
    made.counters_charged = 0
    assert checked(Tool.BUILD_SKETCH, given, made, tmp_path)["valid"] is False


def test_build_sketch_refuses_a_sketch_over_the_memory_left(tmp_path: Path) -> None:
    sketch(tmp_path, 5, 2)
    with pytest.raises(ValueError, match=r"needs 8 counters, but only 6 .* with depth 2, width may be at most 3"):
        sketch(tmp_path, 4, 2)
    with pytest.raises(ValueError, match="no sketch of depth 7 fits"):
        sketch(tmp_path, 1, 7)
    assert len(json.loads((tmp_path / LEDGER).read_text())["sketches"]) == 1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"width": 0}, "width must be at least 1"),
        ({"depth": 0}, "depth must be from 1 to 16"),
        ({"depth": 17}, "depth must be from 1 to 16"),
        ({"min_count": 0}, "min_count must be at least 1"),
        ({"memory_budget": 0}, "memory_budget must be at least 1"),
    ],
)
def test_build_sketch_says_what_is_wrong(tmp_path: Path, changes: dict, message: str) -> None:
    fields = {"width": 2, "depth": 2, "memory_budget": 16} | changes
    with pytest.raises(ValueError, match=message):
        sketch(tmp_path, **fields)


def test_the_ledger_keeps_the_task_fixed(tmp_path: Path) -> None:
    sketch(tmp_path, 4, 2)
    with pytest.raises(ValueError, match="pass memory_budget 16"):
        sketch(tmp_path, 4, 2, memory_budget=17)
    with pytest.raises(ValueError, match="pass min_count 3"):
        sketch(tmp_path, 4, 2, min_count=2)
    count(tmp_path, 1, 1, sketch_id="edge-cases-w4-d2")
    with pytest.raises(ValueError, match="pass query_budget 8"):
        count(tmp_path, 2, 1, sketch_id="edge-cases-w4-d2", query_budget=9)
    with pytest.raises(ValueError, match="pass min_count 3"):
        count(tmp_path, 2, 1, sketch_id="edge-cases-w4-d2", min_count=4)


def test_collisions_inflate_estimates_as_built(tmp_path: Path) -> None:
    sketch(tmp_path, 8, 2)
    saved = json.loads((tmp_path / "candidates" / "edge-cases-w8-d2.json").read_text())
    estimates = {(a, b): estimate for a, b, estimate in saved["candidates"]}
    assert estimates[("delta", "iota")] == 5  # true count 1, collides with (beta, gamma) in both rows
    assert ("eta", "zeta") not in estimates  # true count 1, collides in row 0 only
    assert ("alpha", "gamma") not in estimates  # true count 2 = k - 1
    assert estimates[("alpha", "beta")] >= 3 and estimates[("beta", "gamma")] >= 4


@pytest.fixture
def built(tmp_path: Path) -> Path:
    sketch(tmp_path, 8, 2)
    return tmp_path


def test_count_pairs_counts_exactly_and_charges_only_new_pairs(built: Path) -> None:
    given, made = count(built, 1, 2)
    assert (made.counted, made.skipped, made.total_counted, made.remaining) == (2, 0, 2, 6)
    assert checked(Tool.COUNT_PAIRS, given, made, built)["valid"] is True
    given, made = count(built, 2, 2)
    assert (made.counted, made.skipped, made.total_counted, made.remaining) == (1, 1, 3, 5)
    assert checked(Tool.COUNT_PAIRS, given, made, built)["valid"] is True
    ledger = json.loads((built / LEDGER).read_text())
    found = [pair for call in ledger["counts"] for pair in call["qualifying"]]
    assert sorted((pair["a"], pair["b"], pair["count"]) for pair in found) == [
        ("alpha", "beta", 3),
        ("beta", "gamma", 4),
    ]


def test_count_pairs_refuses_a_range_over_budget_and_says_what_fits(built: Path) -> None:
    with pytest.raises(ValueError, match=r"hold 3 pairs not yet counted \(0 already counted\), but only 2 of the query"
                       r" budget 2 remain: with start_rank 1, count may be at most 2"):
        count(built, 1, 3, query_budget=2)
    assert json.loads((built / LEDGER).read_text())["counts"] == []
    count(built, 1, 2, query_budget=2)
    given, made = count(built, 1, 2, query_budget=2)
    assert (made.counted, made.skipped, made.qualifying) == (0, 2, [])
    assert checked(Tool.COUNT_PAIRS, given, made, built)["valid"] is True
    with pytest.raises(ValueError, match="count may be at most 2; the query budget is used up"):
        count(built, 1, 3, query_budget=2)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"sketch_id": "edge-cases-w4-d4"}, "call build_sketch for it first"),
        ({"sketch_id": "acl-small-w8-d2"}, "is not a edge-cases sketch"),
        ({"sketch_id": "../x"}, "is not a edge-cases sketch"),
        ({"start_rank": 0}, "at least 1"),
        ({"query_budget": 0}, "query_budget must be at least 1"),
        ({"start_rank": 100}, "start_rank must be at most"),
        ({"count": 100}, "count may be at most"),
    ],
)
def test_count_pairs_says_what_is_wrong(built: Path, changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        count(built, 1, 1, **changes)


def test_count_pairs_needs_a_sketch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="call build_sketch first"):
        count(tmp_path, 1, 1)


def test_verify_output_refuses_a_tampered_count(built: Path) -> None:
    given, made = count(built, 1, 3)
    assert made.qualifying
    assert checked(Tool.COUNT_PAIRS, given, made, built)["valid"] is True
    made.remaining += 1
    assert checked(Tool.COUNT_PAIRS, given, made, built)["valid"] is False
    made.remaining -= 1
    made.qualifying[0].count -= 1
    assert checked(Tool.COUNT_PAIRS, given, made, built)["valid"] is False


@pytest.mark.parametrize(("dataset", "min_count", "pairs"), [
    (Dataset.ACL_SMALL, 10, 113), (Dataset.ACL_SMALL, 100, 0), (Dataset.ACL_FULL, 55, 146), (Dataset.EDGE_CASES, 3, 2)
])
def test_the_reference_matches_a_slow_count(tmp_path: Path, dataset: Dataset, min_count: int, pairs: int) -> None:
    given = FindInput(dataset=dataset, min_count=min_count, memory_budget=1, query_budget=1, min_recall="1")
    found = exact_frequent_pairs(given, tmp_path)
    slow = {pair: number for pair, number in slow_counts(dataset).items() if number >= min_count}
    assert {(pair.a, pair.b): pair.count for pair in found.pairs} == slow
    assert len(found.pairs) == pairs
    assert checked(Tool.EXACT_FREQUENT_PAIRS, given, found, tmp_path)["valid"] is True


def test_edge_cases_hold_the_boundaries() -> None:
    counts = slow_counts(Dataset.EDGE_CASES)
    assert counts[("alpha", "beta")] == 3  # exactly k
    assert counts[("alpha", "gamma")] == 2  # k - 1
    assert counts[("beta", "gamma")] == 4  # k + 1, one paper repeating gamma counts once
    assert ("machine translation", "translation") not in counts  # nested, excluded however often it occurs
    corpus = load(Dataset.EDGE_CASES)
    assert sum(corpus.pair_counts.values()) == 14


def test_the_two_rule_files_are_identical() -> None:
    assert (ROOT / "src/plugin/rule.py").read_text() == (ROOT / "verifier/rule.py").read_text()


def test_the_histogram_rule() -> None:
    assert rule.histogram([9, 9, 5, 3, 3, 3], 3) == [(9, 9, 2, 1, 2), (5, 5, 1, 3, 3), (3, 3, 3, 4, 6)]
    estimates = list(range(80, 2, -1))  # 78 distinct values from 80 down to 3: bands of ratio sqrt(2) from 3
    buckets = rule.histogram(estimates, 3)
    assert [(low, high) for low, high, *_ in buckets] == [
        (68, 80), (48, 67), (34, 47), (24, 33), (17, 23), (12, 16), (9, 11), (6, 8), (5, 5), (3, 4)
    ]
    assert buckets[-1][4] == len(estimates) and rule.histogram([], 3) == []
    assert verifier.rule.histogram(estimates, 3) == buckets


@pytest.mark.parametrize("text", ["0", "0.0", "1.001", "1.5", "0.8000", ".8", "0.8e0", " 0.8", "-0.5", "0,8", "nan", ""])
def test_decimals_outside_the_rule_are_refused(text: str) -> None:
    assert rule.parse_decimal(text) is None and verifier.rule.parse_decimal(text) is None


@pytest.mark.parametrize(
    ("dataset", "min_count", "width", "depth", "query", "recall"),
    [
        (Dataset.ACL_FULL, 55, 1500, 2, 400, 142),
        (Dataset.ACL_FULL, 55, 1000, 3, 400, 144),
        (Dataset.ACL_FULL, 55, 3000, 1, 400, 90),
        (Dataset.ACL_FULL, 55, 375, 8, 400, 134),
        (Dataset.ACL_SMALL, 10, 333, 3, 500, 108),
        (Dataset.ACL_SMALL, 10, 1000, 1, 500, 101),
    ],
)
def test_the_instances_separate_settings(
    tmp_path: Path, dataset: Dataset, min_count: int, width: int, depth: int, query: int, recall: int
) -> None:
    """Counting the top query_budget ranks finds these many qualifying pairs, as measured at checkpoint 2."""
    made = build_sketch(SketchInput(dataset, width, depth, min_count, width * depth), tmp_path)
    found = count_pairs(CountInput(dataset, made.sketch_id, 1, min(query, made.candidates), min_count, query), tmp_path)
    assert len(found.qualifying) == recall
