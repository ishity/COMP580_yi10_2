from .checks import HASH_ROWS, result


def verify_input(value):
    return result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    if output_value.dataset != input_value.dataset:
        return result(False, f"output describes {output_value.dataset.value}, not {input_value.dataset.value}")
    if output_value.papers < 1 or output_value.concepts < 2:
        return result(False, "a corpus needs at least 1 paper and 2 concepts")
    if not output_value.distinct_pairs <= output_value.concepts * (output_value.concepts - 1) // 2:
        return result(False, f"{output_value.concepts} concepts make at most {output_value.concepts} choose 2 pairs")
    if output_value.pair_occurrences < output_value.distinct_pairs:
        return result(False, "every distinct pair occurs at least once, so occurrences cannot be fewer")
    if output_value.hash_rows != HASH_ROWS:
        return result(False, f"the plugin bundles {HASH_ROWS} hash rows, not {output_value.hash_rows}")
    return result(True, "the corpus is described", papers=output_value.papers, pairs=output_value.distinct_pairs)
