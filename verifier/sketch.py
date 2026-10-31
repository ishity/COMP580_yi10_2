from pathlib import Path

from .checks import CANDIDATES, HASH_ROWS, LEDGER, candidates_problem, charges_problem, histogram_of, read_json, result


def verify_input(value):
    if value.width < 1 or not 1 <= value.depth <= HASH_ROWS:
        return result(False, f"width must be at least 1 and depth from 1 to {HASH_ROWS}")
    if value.min_count < 1 or value.memory_budget < 1:
        return result(False, "min_count and memory_budget must each be at least 1")
    if value.width * value.depth > value.memory_budget:
        return result(False, f"a {value.width} x {value.depth} sketch exceeds the memory budget {value.memory_budget}")
    return result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    expected = f"{input_value.dataset.value}-w{input_value.width}-d{input_value.depth}"
    if output_value.sketch_id != expected:
        return result(False, f"sketch_id must be {expected}, not {output_value.sketch_id}")
    ledger = read_json(Path(workspace) / LEDGER)
    if ledger is None or not ledger["sketches"]:
        return result(False, "build_sketch left no record of its call in the ledger")
    if (ledger["dataset"], ledger["min_count"], ledger["memory_budget"]) != (
        input_value.dataset.value,
        input_value.min_count,
        input_value.memory_budget,
    ):
        return result(False, "the ledger is for another dataset, min_count or memory_budget than this call")
    problem = charges_problem(ledger)
    if problem:
        return result(False, problem)
    call = ledger["sketches"][-1]
    if call["sketch_id"] != expected:
        return result(False, "the ledger's last sketch is not this call's")
    used = sum(past["charged"] for past in ledger["sketches"])
    if (output_value.counters_charged, output_value.memory_used, output_value.memory_remaining) != (
        call["charged"],
        used,
        input_value.memory_budget - used,
    ):
        return result(False, f"{used} counters are used: counters_charged, memory_used or memory_remaining disagree")
    if used > input_value.memory_budget:
        return result(False, f"{used} counters are used, over the memory budget of {input_value.memory_budget}")
    saved = read_json(Path(workspace) / CANDIDATES / f"{expected}.json")
    if saved is None:
        return result(False, f"sketch {expected}'s candidates are not saved in the workspace")
    problem = candidates_problem(
        saved, expected, input_value.dataset.value, input_value.width, input_value.depth, input_value.min_count
    )
    if problem:
        return result(False, problem)
    if output_value.candidates != len(saved["candidates"]):
        return result(False, f"output counts {output_value.candidates} candidates, the sketch saved {len(saved['candidates'])}")
    given = [
        (row.min_estimate, row.max_estimate, row.candidates, row.first_rank, row.last_rank)
        for row in output_value.histogram
    ]
    if given != histogram_of(saved):
        return result(False, "the histogram does not match the saved candidates")
    return result(True, "the sketch's candidates are ranked and counted", candidates=output_value.candidates)
