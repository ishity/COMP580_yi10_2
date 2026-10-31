"""Each tool's and task's typed input and output (see nanoagent.plugin.v0001.contract): declarations only."""

from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from slimconfig import Output


class Tool(Enum):
    DESCRIBE_CORPUS = "describe_corpus"
    BUILD_SKETCH = "build_sketch"
    COUNT_PAIRS = "count_pairs"
    EXACT_FREQUENT_PAIRS = "exact_frequent_pairs"


class Task(Enum):
    FIND_FREQUENT_PAIRS = "find-frequent-pairs"


class Dataset(Enum):
    ACL_SMALL = "acl-small"
    ACL_FULL = "acl-full"
    EDGE_CASES = "edge-cases"


DATASET = "The bundled corpus of papers, each with its set of concepts"
MIN_COUNT = "The task input's min_count k, copied as given: a pair qualifies when at least k papers hold both concepts"
MEMORY_BUDGET = "The task input's memory_budget, copied as given: the most counters all sketches may use together"
QUERY_BUDGET = "The task input's query_budget, copied as given: the most distinct pairs count_pairs may count"
SKETCH_ID = "The sketch's id, <dataset>-w<width>-d<depth>, as build_sketch returned it"
CONCEPT_A = "The pair's first concept, the one that sorts first"
CONCEPT_B = "The pair's second concept, the one that sorts second"


@dataclass
class Pair(Output):
    a: Annotated[str, CONCEPT_A]
    b: Annotated[str, CONCEPT_B]


@dataclass
class CountedPair(Output):
    a: Annotated[str, CONCEPT_A]
    b: Annotated[str, CONCEPT_B]
    count: Annotated[int, "How many papers hold both concepts, counted exactly"]


@dataclass
class EstimateBucket(Output):
    min_estimate: Annotated[int, "The lowest sketch estimate among this bucket's candidates"]
    max_estimate: Annotated[int, "The highest sketch estimate among this bucket's candidates"]
    candidates: Annotated[int, "How many candidates fall in this bucket"]
    first_rank: Annotated[int, "The rank of this bucket's first candidate, from 1"]
    last_rank: Annotated[int, "The rank of this bucket's last candidate"]


@dataclass
class DescribeInput(Output):
    dataset: Annotated[Dataset, DATASET]


@dataclass
class DescribeOutput(Output):
    dataset: Annotated[Dataset, DATASET]
    papers: Annotated[int, "How many papers the corpus holds"]
    concepts: Annotated[int, "How many concepts the vocabulary holds, V"]
    distinct_pairs: Annotated[int, "How many distinct concept pairs occur together in at least one paper"]
    pair_occurrences: Annotated[int, "How many pairs the stream holds in all, N: each paper's pairs, summed"]
    hash_rows: Annotated[int, "How many fixed hash rows a sketch may use, the largest depth"]


@dataclass
class SketchInput(Output):
    dataset: Annotated[Dataset, DATASET]
    width: Annotated[int, "How many counters each hash row has, at least 1"]
    depth: Annotated[int, "How many hash rows the sketch uses, from 1 to hash_rows"]
    min_count: Annotated[int, MIN_COUNT]
    memory_budget: Annotated[int, MEMORY_BUDGET]


@dataclass
class SketchOutput(Output):
    sketch_id: Annotated[str, SKETCH_ID]
    counters_charged: Annotated[int, "How many counters this call charged: width times depth, or 0 for a sketch built before"]
    memory_used: Annotated[int, "How many counters all sketches built so far use together"]
    memory_remaining: Annotated[int, "How many more counters the memory budget allows"]
    candidates: Annotated[int, "How many pairs of the stream the sketch estimates at min_count or more"]
    histogram: Annotated[
        list[EstimateBucket], "The candidates grouped by sketch estimate, highest first, in rank order"
    ]


@dataclass
class CountInput(Output):
    dataset: Annotated[Dataset, DATASET]
    sketch_id: Annotated[str, SKETCH_ID]
    start_rank: Annotated[int, "The rank of the first candidate to count, from 1"]
    count: Annotated[int, "How many candidates to count from start_rank on, at least 1"]
    min_count: Annotated[int, MIN_COUNT]
    query_budget: Annotated[int, QUERY_BUDGET]


@dataclass
class CountOutput(Output):
    qualifying: Annotated[list[CountedPair], "The newly counted pairs in the range that qualify, in rank order"]
    counted: Annotated[int, "How many pairs in the range were counted now and charged to the query budget"]
    skipped: Annotated[int, "How many pairs in the range an earlier call counted, so not counted or charged again"]
    total_counted: Annotated[int, "How many distinct pairs all calls so far have counted"]
    remaining: Annotated[int, "How many more pairs the query budget allows"]


@dataclass
class FindInput(Output):
    dataset: Annotated[Dataset, DATASET]
    min_count: Annotated[int, "The least number of papers a pair must appear in together, k"]
    memory_budget: Annotated[int, "The most counters all sketches may use together"]
    query_budget: Annotated[int, "The most distinct pairs count_pairs may count, over all its calls"]
    min_recall: Annotated[str, "The least share of all qualifying pairs to find, as a decimal string, e.g. 0.90"]


@dataclass
class ExactOutput(Output):
    dataset: Annotated[Dataset, DATASET]
    min_count: Annotated[int, "The least number of papers a pair must appear in together, k"]
    concepts: Annotated[list[str], "Every concept of the dataset, in its order"]
    pairs: Annotated[list[CountedPair], "Every qualifying pair, counted exactly, sorted by a then b"]


@dataclass
class FindOutput(Output):
    pairs: Annotated[list[Pair], "Every qualifying pair count_pairs returned, each once"]


TOOLS = {
    Tool.DESCRIBE_CORPUS: (DescribeInput, DescribeOutput),
    Tool.BUILD_SKETCH: (SketchInput, SketchOutput),
    Tool.COUNT_PAIRS: (CountInput, CountOutput),
    Tool.EXACT_FREQUENT_PAIRS: (FindInput, ExactOutput),
}
TASKS = {Task.FIND_FREQUENT_PAIRS: (FindInput, FindOutput)}
