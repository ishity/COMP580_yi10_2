"""Decimal strings and the histogram rule, in integer and Fraction arithmetic only.

src/plugin/rule.py and verifier/rule.py hold this same file, so the tools and the verifier decide alike.
"""

import re
from fractions import Fraction

DECIMAL = re.compile(r"^[0-9]+(?:\.[0-9]{1,3})?$")
DECIMAL_RULE = 'a decimal string in (0, 1] with at most 3 decimal places, e.g. "0.80"'
EXACT_BUCKETS = 16  # at most this many distinct estimates get one bucket each


def parse_decimal(text):
    """``text`` as a Fraction, or None when it is not DECIMAL_RULE."""
    if not DECIMAL.fullmatch(text):
        return None
    value = Fraction(text)
    return value if 0 < value <= 1 else None


def band(estimate, min_count):
    """The largest j with estimate >= min_count * sqrt(2)^j, decided as estimate^2 >= min_count^2 * 2^j."""
    j = 0
    while estimate * estimate >= min_count * min_count * 2 ** (j + 1):
        j += 1
    return j


def histogram(estimates, min_count):
    """Buckets of ranked estimates (highest first, each at least min_count) as
    (min_estimate, max_estimate, candidates, first_rank, last_rank): one per distinct estimate when there are
    at most EXACT_BUCKETS of them, else one per band of ratio sqrt(2) from min_count."""
    if len(set(estimates)) <= EXACT_BUCKETS:
        keys = list(estimates)
    else:
        keys = [band(estimate, min_count) for estimate in estimates]
    buckets, start = [], 0
    for rank in range(1, len(estimates) + 1):
        if rank == len(estimates) or keys[rank] != keys[start]:
            group = estimates[start:rank]
            buckets.append((min(group), max(group), len(group), start + 1, rank))
            start = rank
    return buckets
