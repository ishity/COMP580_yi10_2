"""The find-frequent-pairs task's output check, against the exact_frequent_pairs reference and the ledger.

The first failing check decides: malformed pairs, the budgets, precision, provenance, then recall.
"""

from fractions import Fraction
from pathlib import Path

from .checks import LEDGER, charges_problem, read_json, result
from .rule import parse_decimal


def verify_task(arguments, output, reference, workspace):
    """``arguments``: a contract FindInput; ``output``: a FindOutput; ``reference``: exact_frequent_pairs' ExactOutput."""
    if (reference.dataset, reference.min_count) != (arguments.dataset, arguments.min_count):
        return result(False, "the reference is for another dataset or min_count than the task")
    min_recall = parse_decimal(arguments.min_recall)
    if min_recall is None:
        return result(False, "min_recall must be a decimal string in (0, 1]")

    # 1. Malformed: two distinct known concepts per pair, each pair once in either order.
    concepts = set(reference.concepts)
    submitted = []
    for pair in output.pairs:
        if pair.a == pair.b:
            return result(False, f"pair ({pair.a!r}, {pair.b!r}) pairs a concept with itself")
        for concept in (pair.a, pair.b):
            if concept not in concepts:
                return result(False, f"{concept!r} is not a concept of {arguments.dataset.value}")
        submitted.append((min(pair.a, pair.b), max(pair.a, pair.b)))
    if len(set(submitted)) != len(submitted):
        return result(False, "a pair is submitted twice, counting (a, b) and (b, a) as one pair")

    # 2. Budgets: the ledger is this task's, its sketches fit the memory budget and its counts the query budget.
    ledger = read_json(Path(workspace) / LEDGER)
    if ledger is None and submitted:
        return result(False, f"there is no {LEDGER}, so no submitted pair was counted")
    sketches = [] if ledger is None else ledger["sketches"]
    calls = [] if ledger is None else ledger["counts"]
    if ledger is not None:
        if (ledger["dataset"], ledger["min_count"], ledger["memory_budget"]) != (
            arguments.dataset.value,
            arguments.min_count,
            arguments.memory_budget,
        ) or ledger["query_budget"] not in (None, arguments.query_budget):
            return result(False, "the ledger is for another dataset, min_count, memory_budget or query_budget than the task")
        if calls and ledger["query_budget"] is None:
            return result(False, "the ledger counts pairs without a query budget")
        problem = charges_problem(ledger)
        if problem:
            return result(False, problem)
    memory = sum(sketch["charged"] for sketch in sketches)
    if memory > arguments.memory_budget:
        return result(False, f"the sketches use {memory} counters, over the memory budget of {arguments.memory_budget}")
    counted = [tuple(pair) for call in calls for pair in call["counted"]]
    if len(set(counted)) != len(counted):
        return result(False, "the ledger counts a pair twice")
    if len(counted) > arguments.query_budget:
        return result(False, f"{len(counted)} distinct pairs were counted, over the query budget of {arguments.query_budget}")

    # 3. Precision: every submitted pair qualifies by the reference.
    truth = {(pair.a, pair.b): pair.count for pair in reference.pairs}
    false = [pair for pair in submitted if pair not in truth]
    if false:
        return result(False, f"{len(false)} submitted pairs do not qualify, e.g. {false[0]}", false_pairs=len(false))

    # 4. Provenance: count_pairs returned every submitted pair as qualifying, with the reference's exact count.
    returned = {(pair["a"], pair["b"]): pair["count"] for call in calls for pair in call["qualifying"]}
    for pair in submitted:
        if pair not in returned:
            return result(False, f"pair {pair} was never returned as qualifying by count_pairs")
        if returned[pair] != truth[pair]:
            return result(False, f"pair {pair} was recorded with count {returned[pair]}, not its exact {truth[pair]}")

    # 5. Recall.
    recall = Fraction(len(submitted), len(truth)) if truth else Fraction(1)
    details = {"submitted": len(submitted), "reference": len(truth), "counters": memory, "counted": len(counted)}
    message = f"found {len(submitted)} of {len(truth)} qualifying pairs, with {memory} counters and {len(counted)} counted pairs"
    if recall < min_recall:
        return result(False, f"{message}: recall {float(recall):.3f} is below {arguments.min_recall}", score=float(recall), **details)
    return result(True, message, score=float(recall), **details)
