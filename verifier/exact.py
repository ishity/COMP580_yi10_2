from .checks import decimal_problem, result


def verify_input(value):
    problem = decimal_problem(value.min_recall, "min_recall")
    if problem:
        return result(False, problem)
    if value.min_count < 1 or value.memory_budget < 1 or value.query_budget < 1:
        return result(False, "min_count, memory_budget and query_budget must each be at least 1")
    return result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    if (output_value.dataset, output_value.min_count) != (input_value.dataset, input_value.min_count):
        return result(False, "output is for another dataset or min_count than the input")
    concepts = set(output_value.concepts)
    if len(concepts) != len(output_value.concepts) or len(concepts) < 2:
        return result(False, "the concepts must be at least 2 and unique")
    pairs = [(pair.a, pair.b) for pair in output_value.pairs]
    if pairs != sorted(set(pairs)):
        return result(False, "the pairs must be unique and sorted by a then b")
    for pair in output_value.pairs:
        if pair.a not in concepts or pair.b not in concepts:
            return result(False, f"pair ({pair.a!r}, {pair.b!r}) names a concept the dataset does not hold")
        if not pair.a < pair.b:
            return result(False, f"pair ({pair.a!r}, {pair.b!r}) is not in canonical order, a before b")
        if pair.count < input_value.min_count:
            return result(False, f"pair ({pair.a!r}, {pair.b!r}) counts {pair.count}, below min_count")
    return result(True, "every qualifying pair is listed", pairs=len(pairs))
