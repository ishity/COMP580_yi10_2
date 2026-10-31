"""describe_corpus: how big a bundled corpus and its pair stream are, and how many hash rows a sketch may use."""

from __future__ import annotations

from pathlib import Path

from contract import DescribeInput, DescribeOutput
from plugin.bundle import hash_rows, load


def describe_corpus(given: DescribeInput, workspace: Path) -> DescribeOutput:
    corpus = load(given.dataset)
    return DescribeOutput(
        dataset=given.dataset,
        papers=corpus.papers,
        concepts=len(corpus.concepts),
        distinct_pairs=len(corpus.pair_counts),
        pair_occurrences=sum(corpus.pair_counts.values()),
        hash_rows=len(hash_rows()[1]),
    )
