"""Checks shared by the tool verifiers. Pure functions of their arguments and the files on disk."""

import json
import re
from pathlib import Path

from .rule import DECIMAL_RULE, histogram, instance_name, parse_decimal

LEDGERS = "ledgers"
CANDIDATES = "candidates"
HASH_ROWS = 16  # how many rows src/plugin/data/hashes.json holds


def result(valid, message, *, score=None, **details):
    return {
        "valid": valid,
        "score": float(valid) if score is None else score,
        "message": message,
        "details": details,
    }


def decimal_problem(text, name):
    if parse_decimal(text) is None:
        return f"{name} must be {DECIMAL_RULE}, got {text!r}"
    return None


def sketch_id_problem(dataset, sketch_id):
    if not re.fullmatch(re.escape(dataset.value) + r"-w[1-9][0-9]*-d[1-9][0-9]*", sketch_id):
        return f"sketch_id {sketch_id!r} is not a {dataset.value} sketch, {dataset.value}-w<width>-d<depth>"
    return None


def instance_of(value):
    """The instance name of a task input or tool input: its dataset, min_count and both budgets."""
    return instance_name(value.dataset.value, value.min_count, value.memory_budget, value.query_budget)


def ledger_file(workspace, name):
    return Path(workspace) / LEDGERS / f"{name}.json"


def candidates_file(workspace, name, sketch_id):
    return Path(workspace) / CANDIDATES / name / f"{sketch_id}.json"


def ledger_problem(ledger, value):
    """What is wrong when a ledger's locked values are not ``value``'s, or None."""
    if (ledger["dataset"], ledger["min_count"], ledger["memory_budget"], ledger["query_budget"]) != (
        value.dataset.value,
        value.min_count,
        value.memory_budget,
        value.query_budget,
    ):
        return "the ledger is for another dataset, min_count, memory_budget or query_budget than the task"
    return None


def read_json(path):
    """The JSON in ``path``, or None when there is no such file."""
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def charges_problem(ledger):
    """What is wrong with the ledger's sketch calls: each id is charged width x depth once, then 0."""
    built = set()
    for call in ledger["sketches"]:
        if call["sketch_id"] != f"{ledger['dataset']}-w{call['width']}-d{call['depth']}":
            return f"the ledger names sketch {call['sketch_id']} for another width or depth"
        expected = 0 if call["sketch_id"] in built else call["width"] * call["depth"]
        if call["charged"] != expected:
            return f"the ledger charges sketch {call['sketch_id']} {call['charged']} counters, not {expected}"
        built.add(call["sketch_id"])
    return None


def candidates_problem(saved, sketch_id, dataset, width, depth, min_count):
    """What is wrong with a saved sketch's ranked candidates, or None."""
    if (saved["sketch_id"], saved["dataset"], saved["width"], saved["depth"], saved["min_count"]) != (
        sketch_id, dataset, width, depth, min_count
    ):
        return "the saved candidates are not for the requested sketch and min_count"
    pairs = [(a, b) for a, b, _ in saved["candidates"]]
    if len(set(pairs)) != len(pairs):
        return "the saved sketch lists a candidate pair twice"
    for a, b, estimate in saved["candidates"]:
        if not a < b:
            return f"candidate ({a!r}, {b!r}) is not in canonical order, a before b"
        if estimate < min_count:
            return f"candidate ({a!r}, {b!r}) is estimated at {estimate}, below min_count {min_count}"
    keys = [(-estimate, a, b) for a, b, estimate in saved["candidates"]]
    if keys != sorted(keys):
        return "the saved candidates are not ranked by estimate, highest first, then by (a, b)"
    return None


def histogram_of(saved):
    return histogram([estimate for _, _, estimate in saved["candidates"]], saved["min_count"])
