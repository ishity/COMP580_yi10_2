"""The bundled corpora and hash rows, and the names and paths the tools share."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from itertools import combinations
from pathlib import Path

from contract import Dataset
from plugin.rule import DECIMAL_RULE, instance_name, parse_decimal

DATA = Path(__file__).with_name("data")
LEDGERS = "ledgers"
CANDIDATES = "candidates"


@dataclass(frozen=True)
class Corpus:
    concepts: tuple[str, ...]
    papers: int
    # Each concept pair occurring together, as indices i < j, and how many papers hold both.
    pair_counts: dict[tuple[int, int], int]


@cache
def load(dataset: Dataset) -> Corpus:
    """``dataset``'s concepts and its pair stream: each paper's distinct concepts, every pair of them except the
    excluded (nested) ones, a paper repeating a concept counting it once."""
    raw = json.loads((DATA / f"{dataset.value}.json").read_text())
    concepts = raw["keywords"]
    if len(set(concepts)) != len(concepts):
        raise ValueError(f"{dataset.value} must hold unique concepts")
    excluded = {(i, j) for i, j in raw["excluded_pairs"]}
    counts: Counter[tuple[int, int]] = Counter()
    for paper in raw["papers"]:
        held = sorted(set(paper["concepts"]))
        if held and not 0 <= held[0] <= held[-1] < len(concepts):
            raise ValueError(f"{dataset.value} paper {paper['id']} names a concept index out of range")
        counts.update(pair for pair in combinations(held, 2) if pair not in excluded)
    return Corpus(concepts=tuple(concepts), papers=len(raw["papers"]), pair_counts=dict(counts))


@cache
def hash_rows() -> tuple[int, tuple[tuple[int, int], ...]]:
    """The prime p and each row's (a, b), for h(x) = ((a * x + b) mod p) mod width."""
    raw = json.loads((DATA / "hashes.json").read_text())
    return raw["prime"], tuple((row["a"], row["b"]) for row in raw["rows"])


def names(corpus: Corpus, pair: tuple[int, int]) -> tuple[str, str]:
    """A pair of concept indices as concept names in canonical order: the name that sorts first, then the other."""
    a, b = corpus.concepts[pair[0]], corpus.concepts[pair[1]]
    return (a, b) if a < b else (b, a)


def decimal(text: str, name: str) -> Fraction:
    value = parse_decimal(text)
    if value is None:
        raise ValueError(f"{name} must be {DECIMAL_RULE}, got {text!r}")
    return value


def sketch_id(dataset: Dataset, width: int, depth: int) -> str:
    return f"{dataset.value}-w{width}-d{depth}"


def sketch_id_pattern(dataset: Dataset) -> str:
    return re.escape(dataset.value) + r"-w[1-9][0-9]*-d[1-9][0-9]*"


def instance(dataset: Dataset, min_count: int, memory_budget: int, query_budget: int) -> str:
    """The task instance's name, which names its ledger and its candidate folder."""
    return instance_name(dataset.value, min_count, memory_budget, query_budget)


def candidates_path(workspace: Path, name: str, sketch: str) -> Path:
    return workspace / CANDIDATES / name / f"{sketch}.json"


def ledger_path(workspace: Path, name: str) -> Path:
    return workspace / LEDGERS / f"{name}.json"


def read_ledger(workspace: Path, name: str) -> dict | None:
    path = ledger_path(workspace, name)
    return json.loads(path.read_text()) if path.exists() else None


def write_ledger(workspace: Path, name: str, ledger: dict) -> None:
    path = ledger_path(workspace, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger) + "\n")
