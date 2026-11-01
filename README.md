# Concept Co-occurrence Query

A NanoAgent plugin with one checkable job: given a bundled corpus of ACL Anthology papers, each with a
set of concepts, and a threshold k, find every concept pair that appears together in at least k papers.
A Count-Min Sketch, limited to a memory budget of counters, proposes candidates; exact counting, limited
to a query budget of distinct pairs, confirms them. The plugin is a standard MCP server over stdio;
NanoAgent owns its launch command, environment, input and output checks and lifecycle.

## Tools

| Tool | Does | Code |
|---|---|---|
| `describe_corpus` | Gives a corpus's paper count, concept count V, distinct co-occurring pairs, total pair occurrences N and the hash rows a sketch may use (16) | `src/plugin/describe.py` |
| `build_sketch` | Streams every pair occurrence into a `depth × width` Count-Min Sketch with the first `depth` hash rows, charges `width × depth` counters to the memory budget (once per sketch id), estimates every pair of the stream, saves those estimated at `min_count` or more to `candidates/<instance>/<sketch_id>.json` in the workspace, ranked by estimate, and returns the sketch id, the counters charged and left, the candidate count and a histogram by estimate | `src/plugin/sketch.py` |
| `count_pairs` | Counts a rank range of a saved sketch exactly and returns the qualifying pairs with their counts, including skipped pairs an earlier call found qualifying, marked `previously_counted`; records every call in the instance's ledger and refuses a range that would take the distinct counted pairs past the query budget | `src/plugin/count.py` |
| `exact_frequent_pairs` | Counts every pair exactly and lists those in at least k papers: the task's reference, which the model never sees | `src/plugin/exact.py` |

### Tasks

| Task | Does | Runner |
|---|---|---|
| `find-frequent-pairs` | Finds every concept pair appearing together in at least `min_count` papers, within a memory budget and a query budget | `agent`, with `describe_corpus`, `build_sketch` and `count_pairs`; reference `exact_frequent_pairs` |

The task input is a `dataset`, `min_count` (k), `memory_budget` (the most counters all sketches may use
together), `query_budget` (the most distinct pairs `count_pairs` may count) and `min_recall` (a decimal
string such as `"0.90"`, in (0, 1] with at most 3 decimal places). The output is the list of qualifying
pairs, each as two concept names.

The pair stream holds, for each paper, every pair of its distinct concepts (a concept a paper lists
twice counts once), except nested pairs, where one concept's phrase lies inside the other's (such as
`machine translation` and `neural machine translation`). A pair of concept indices `i < j` has the key
`x = i · V + j`, and hash row `(a, b)` sends it to counter `((a · x + b) mod p) mod width`, with
`p = 2³¹ − 1`. A pair's estimate is the least of its `depth` counters, never below its true count, so
every qualifying pair is a candidate. Candidates are the stream's pairs estimated at `min_count` or more
(the two-pass Count-Min heavy-hitter method), ranked by estimate, ties by name.

Candidate lists never pass through the model: the agent sees each sketch's histogram and asks
`count_pairs` for rank ranges. The histogram has one bucket per distinct estimate when there are at most
16, else one per band of ratio √2 from k, with integer edges (`e² ≥ k² · 2ʲ`). Both tools take the
task's dataset, `min_count`, `memory_budget` and `query_budget`, and these four values name the instance's
ledger, `ledgers/<dataset>-k<min_count>-m<memory_budget>-q<query_budget>.json` (for example
`ledgers/acl-small-k10-m1000-q500.json`), and its candidate folder `candidates/<instance>/`. All four are
integers; `min_recall`, the only decimal, is never passed to a tool, so it is not part of the name. The
workspace outlives a run, so different instances in one workspace never share a ledger, and a second run of
the same instance continues the first's: rebuilding a sketch with the same width and depth is not charged
again, and a pair any earlier call counted is skipped, not charged again, but returned once more with its
recorded count and `previously_counted` set if it qualified. Every refused call says what was wrong and
which value would be valid, for example the largest width or `count` that still fits.

`verify_task` checks, in order, and the first failure decides:

1. Malformed: every pair names two distinct concepts of the dataset, and no pair is submitted twice,
   counting `(a, b)` and `(b, a)` as one.
2. Budgets: the ledger named by the task's dataset, `min_count` and both budgets holds those values, each
   sketch is charged
   `width × depth` once, the counters stay within the memory budget and the distinct counted pairs
   within the query budget. A non-empty answer without a ledger fails.
3. Precision: every submitted pair is in the reference. Any false pair scores 0.
4. Provenance: `count_pairs` returned every submitted pair as qualifying, with the reference's exact count,
   in this run or an earlier run of the same instance.
5. Recall: `recall = |submitted| / |reference|` (1 for an empty answer to an empty reference). The
   result is valid when recall is at least `min_recall`, and its score is the recall.

The bundled instances:

| Dataset | Papers | Distinct pairs | k | Qualifying | Memory budget | Query budget | min_recall |
|---|---|---|---|---|---|---|---|
| `acl-small` | 2,924 | 2,597 | 10 | 113 | 1,000 | 500 | 0.90 |
| `acl-full` | 21,934 | 11,009 | 55 | 146 | 3,000 | 400 | 0.95 |
| `acl-small` | 2,924 | 2,597 | 100 | 0 | 500 | 100 | 1.0 |
| `edge-cases` | 17 | 8 | 3 | 2 | 16 | 8 | 1.0 |

The memory budgets stay well below the distinct pairs, so an exact table does not fit. `acl-full` and
`acl-small` come from ACL Anthology papers of 2016 to 2024 (2018 and 2019 for `acl-small`) with an
abstract: a paper's concepts are the 493 keywords its title and abstract mention as whole phrases,
after lowercasing, mapping every character outside `a-z0-9` to a space and dropping a final `s` from
tokens longer than 3 characters not ending in `ss`, `us` or `is`. Papers with fewer than 2 concepts are
dropped. `edge-cases` holds pairs with counts exactly k, k − 1 and k + 1, a paper repeating a concept, an
excluded nested pair, and two once-seen pairs that collide with a frequent pair in the bundled hash
rows. The generator lives outside this repository; only its JSON output is here. Source: ACL Anthology,
https://aclanthology.org/, CC BY 4.0; the data holds anthology ids, years and concept indices only.

## Layout

It has the standard plugin layout, which NanoAgent requires:

```text
.
├── nanoagent.yaml   # manifest (a slimconfig ManifestConfig): id, runtime, permissions, tools, tasks
├── contract.py      # each tool's and task's typed input and output: slimconfig Output classes
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env             # needed environment variable names (none here)
├── src/plugin/      # the MCP server (__main__.py, its IMPLEMENTATIONS table), one module per tool,
│                    # bundle.py (corpora, hash rows and paths), rule.py (decimals, the histogram rule, instance names)
│                    # and data/ (the JSON)
├── tests/           # test_tools.py and test_verifier.py
├── ui/              # renderers.js and renderers.css: outputs are drawn from their JSON Schema
├── product/         # the product page a catalog shows: product.yaml and icon.svg
└── verifier/        # deterministic verifier package, one module per tool, find.py for the task
```

`contract.py` types every value: the server builds each call's input into its class and checks its
output against its own, and NanoAgent and the verifier get the same classes. A failing tool raises,
and the MCP server returns an error result.

`verifier/` is deterministic: it imports only the standard library and `contract`, and reads only
the workspace's `candidates/` and `ledgers/`. Beyond the task check, it rederives every tool call's
meaning: for `build_sketch`, the sketch id, each sketch's charge, the counters used and left, and the
ranking and histogram against the saved candidates; for each `count_pairs` call, that the ledger counted
exactly the range's new pairs, that every qualifying pair reaches `min_count`, that the skipped pairs
returned as previously counted are exactly those an earlier call recorded as qualifying, with the same
counts, and that the totals and remaining budget add up. `src/plugin/rule.py` and `verifier/rule.py` hold the same file, word for word.

## Develop and run

```bash
export UV_NO_CONFIG=1   # ignore user and system uv.toml, such as an exclude-newer setting
uv lock
uv sync --extra dev
uv run pytest
uv run ruff check .
uv lock --check
UV_CACHE_DIR=$(mktemp -d) uv lock --check --offline
```

NanoAgent passes the folder the plugin may write to as `NANOAGENT_WORKSPACE`; `build_sketch` writes
`candidates/<instance>/<sketch_id>.json` there, and both `build_sketch` and `count_pairs` write
`ledgers/<instance>.json`. RUSH Lab copies a member's files into every run's workspace and writes them back
afterwards, so these files persist from run to run; the per-instance names keep the instances apart.
