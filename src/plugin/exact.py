"""exact_frequent_pairs: every qualifying pair of a corpus, counted exactly, the task's reference."""

from __future__ import annotations

from pathlib import Path

from contract import CountedPair, ExactOutput, FindInput
from plugin.bundle import load, names


def exact_frequent_pairs(given: FindInput, workspace: Path) -> ExactOutput:
    if given.min_count < 1:
        raise ValueError(f"min_count must be at least 1, got {given.min_count}")
    corpus = load(given.dataset)
    pairs = [
        CountedPair(*names(corpus, pair), count=count)
        for pair, count in corpus.pair_counts.items()
        if count >= given.min_count
    ]
    return ExactOutput(
        dataset=given.dataset,
        min_count=given.min_count,
        concepts=list(corpus.concepts),
        pairs=sorted(pairs, key=lambda pair: (pair.a, pair.b)),
    )
