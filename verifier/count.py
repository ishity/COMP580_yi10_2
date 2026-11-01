from .checks import candidates_file, instance_of, ledger_file, ledger_problem, read_json, result, sketch_id_problem


def verify_input(value):
    problem = sketch_id_problem(value.dataset, value.sketch_id)
    if problem:
        return result(False, problem)
    if value.start_rank < 1 or value.count < 1:
        return result(False, "start_rank and count must each be at least 1")
    if value.min_count < 1 or value.memory_budget < 1 or value.query_budget < 1:
        return result(False, "min_count, memory_budget and query_budget must each be at least 1")
    return result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    name = instance_of(input_value)
    ledger = read_json(ledger_file(workspace, name))
    if ledger is None or not ledger["counts"]:
        return result(False, "count_pairs left no record of its call in the ledger")
    problem = ledger_problem(ledger, input_value)
    if problem:
        return result(False, problem)
    if all(sketch["sketch_id"] != input_value.sketch_id for sketch in ledger["sketches"]):
        return result(False, f"sketch {input_value.sketch_id} was never built")
    *earlier, call = ledger["counts"]
    if (call["sketch_id"], call["start_rank"], call["count"]) != (
        input_value.sketch_id,
        input_value.start_rank,
        input_value.count,
    ):
        return result(False, "the ledger's last count is not this call's sketch and rank range")
    saved = read_json(candidates_file(workspace, name, input_value.sketch_id))
    if saved is None:
        return result(False, f"sketch {input_value.sketch_id}'s candidates are not saved in the workspace")
    first = input_value.start_rank
    span = [(a, b) for a, b, _ in saved["candidates"][first - 1 : first - 1 + input_value.count]]
    if len(span) != input_value.count:
        return result(False, "the rank range goes past the sketch's candidates")
    seen = {(a, b) for past in earlier for a, b in past["counted"]}
    new = [pair for pair in span if pair not in seen]
    if [tuple(pair) for pair in call["counted"]] != new or call["skipped"] != len(span) - len(new):
        return result(False, "the ledger does not count exactly the range's pairs no earlier call counted")
    if (output_value.counted, output_value.skipped) != (len(new), len(span) - len(new)):
        return result(False, f"of the range's {len(span)} pairs, {len(new)} are new: counted and skipped disagree")
    order = {pair: rank for rank, pair in enumerate(new)}
    ranks = [order.get((pair["a"], pair["b"])) for pair in call["qualifying"]]
    if None in ranks or ranks != sorted(set(ranks)):
        return result(False, "the ledger's qualifying pairs are not newly counted pairs of the range, in rank order")
    for pair in call["qualifying"]:
        if pair["count"] < input_value.min_count:
            return result(False, f"pair ({pair['a']!r}, {pair['b']!r}) counts {pair['count']}, below min_count")
    # The range's qualifying pairs: those counted now, as the ledger records them, and the skipped ones an
    # earlier call recorded as qualifying, with the same counts, all in rank order.
    now = {(pair["a"], pair["b"]): pair["count"] for pair in call["qualifying"]}
    before = {(pair["a"], pair["b"]): pair["count"] for past in earlier for pair in past["qualifying"]}
    expected = [
        {"a": a, "b": b, "count": now[a, b], "previously_counted": False} if (a, b) in now else
        {"a": a, "b": b, "count": before[a, b], "previously_counted": True}
        for a, b in span
        if (a, b) in now or (a, b) in before
    ]
    qualifying = [
        {"a": pair.a, "b": pair.b, "count": pair.count, "previously_counted": pair.previously_counted}
        for pair in output_value.qualifying
    ]
    if qualifying != expected:
        return result(False, "the qualifying pairs differ from the ledger's, counted now or by an earlier call")
    total = len(seen) + len(new)
    if output_value.total_counted != total or output_value.remaining != input_value.query_budget - total:
        return result(False, f"{total} distinct pairs are counted: total_counted or remaining disagree")
    if total > input_value.query_budget:
        return result(False, f"{total} distinct pairs are counted, over the query budget of {input_value.query_budget}")
    return result(True, "the range is counted within budget", counted=len(new), qualifying=len(qualifying), previously_counted=len(qualifying) - len(now))
