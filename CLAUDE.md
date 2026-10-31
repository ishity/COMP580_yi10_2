# CLAUDE.md: build the `concept-cooccurrence-query` NanoAgent plugin

## Context

This is plugin 2 of 5 for the COMP 480/580 milestone "Verifiable Tasks" (due 2026-11-02). Plugin 1 is finished and published at https://github.com/ishity/lsh-similar-pairs (local copy: `../lsh-similar-pairs`). Its design was reviewed in detail and works: reuse its structure, patterns and helpers wherever they fit.

Goal: an original task, **finding every concept pair that co-occurs in at least k papers, using a Count-Min Sketch under a memory budget and exact counting under a query budget**, checked by a deterministic verifier against an exact reference.

The plugin's theme (concept co-occurrence across papers) matches the description registered for the course. The corpus built here will be reused by plugins 4 (concept-gap-detection) and 5 (gap-backtest), so design it as a shared asset.

## Read first

1. `../nanoagent/docs/instructions/skills/publish-plugin/SKILL.md` (authoritative rules; the clone already exists).
2. All of `../lsh-similar-pairs`: `CLAUDE.md`, `contract.py`, `nanoagent.yaml`, `src/plugin/`, `verifier/`, `tests/`, and `../lsh-report-notes.md`. The design decisions recorded there (ledger locking, dedup of repeated items, rank ranges instead of item lists, informative errors, decimal strings parsed with Fraction, candidate files in the workspace, reference tool taking the full task input) all apply here too.

If anything here conflicts with SKILL.md, SKILL.md wins. Tell me about the conflict.

## Hard rules

Identical to plugin 1. In short:
- **uv only.** No pip, conda or pixi. Keep `requires-python`, dependencies and the `slimconfig` source unchanged except for the package-data setting plugin 1 added.
- **Verifier**: pure standard library plus `contract`; no LLM, clock, randomness, environment, network, subprocess, `open`, `eval`, `exec`, `getattr`, private attributes or writes. `Path(...).read_text()` only. Deterministic across two runs.
- **Rules in code, judgment to the model. No fallbacks. Only allowed top-level files.**
- **Item lists never pass through the model.** Candidates live in workspace files; the model works with ids, rank ranges and histograms.
- **Ask me before** creating a GitHub repo, making anything public, or pushing.
- No em-dashes in any text you write.

## Setup

```bash
cd ~
rsync -a --exclude .git --exclude .venv --exclude CLAUDE.md lsh-similar-pairs/ concept-cooccurrence-query/
cd concept-cooccurrence-query
git init -b main
```
Then this file becomes the repo's `CLAUDE.md`. Rename the plugin everywhere (`nanoagent.yaml` id and name, README title, `pyproject.toml` name `nanoagent-concept-cooccurrence-query`, LICENSE stays "Yuri", 2026), version 0.1.0, `status: in-development`. Remove every LSH-specific tool, task, test, verifier module and data file.

## Shared corpus (built outside the repo)

Create `../gapmap-corpus-gen/` as its own uv project.

- Source: ACL Anthology papers with titles and abstracts (for example via the `acl-anthology` Python package or the anthology's public data). Check and record the license of the metadata and abstracts in its README.
- Years: a contiguous range of about 10 years ending no later than 2024, so plugin 5 can split by year.
- Vocabulary: the 493 keywords of `../lsh-data-gen/keywords.txt`.
- Matching rule (deterministic, documented): lowercase, normalize whitespace and hyphens, match each keyword as a whole phrase in title + abstract. A paper's concept set is the set of matched keywords. Drop papers with fewer than 2 concepts.
- Output a master file `corpus.json` with, per paper: a stable paper id (anthology id), year, and the sorted list of keyword indices. Keep abstracts out of it.
- Record the exact command, package versions and the resulting counts in `../gapmap-corpus-gen/README.md`.

From the master corpus, derive this plugin's bundled datasets in `src/plugin/data/`:
- `acl-small`: a deterministic subset (for example every paper of 2 chosen years).
- `acl-full`: all papers.
- `edge-cases`: a hand-built tiny corpus with pairs whose count is exactly k, k - 1 and k + 1, a paper repeating the same keyword (count once), papers with exactly 2 concepts, and hash collisions forced by construction.
- `hashes.json`: fixed parameters for up to 16 hash rows, generated once with a seed: for row i, integers a_i, b_i for `h_i(x) = ((a_i * x + b_i) mod p) mod width` with a fixed prime p larger than any pair key. Pair key: `x = a * V + b` for keyword indices a < b and vocabulary size V.

## Task design

### The task in one sentence

Given a bundled corpus of papers, each with a set of concepts, and a threshold k, find every concept pair that appears together in at least k papers. A Count-Min Sketch, limited to a memory budget of counters, proposes candidates; exact counting, limited to a query budget of distinct pairs, confirms them.

### Why it is objectively verifiable

The answer is a set fully determined by the bundled corpus and k, computed with integer counts only. Precision, recall and both budgets are checkable by code. A Count-Min Sketch never underestimates, so with enough query budget full recall is reachable; any miss comes from how the agent spends its budgets.

### Tools

| Tool | Input | Output | Notes |
|---|---|---|---|
| `describe_corpus` | `dataset` | paper count, concept count (V), distinct co-occurring pair count, total pair occurrences in the stream, hash rows available | |
| `build_sketch` | `dataset`, `width`, `depth`, `min_count`, `memory_budget` | `sketch_id`, counters used, memory remaining, candidate count (pairs whose estimate is at least `min_count`), histogram of candidates by estimate | Streams every paper's concept pairs into a `depth × width` Count-Min Sketch using the first `depth` hash rows. Charges `width × depth` counters against the memory budget (ledger, locked on first call). Saves candidates to `workspace/candidates/<sketch_id>.json`, ranked by estimate descending, ties by (a, b) string order. Histogram buckets: one per distinct estimate value, or bucketed if there are too many; propose a bucketing and give first_rank/last_rank per bucket. |
| `count_pairs` | `dataset`, `sketch_id`, `start_rank`, `count`, `min_count`, `query_budget` | qualifying pairs with their exact counts, counted, skipped, total counted, remaining | Exact counts for a rank range. Pairs counted by any earlier call are skipped and not charged. Refuses, without counting, a range whose new pairs exceed the remaining budget, saying how many new pairs it holds and the largest count that fits. |
| `exact_frequent_pairs` | full task input | all qualifying pairs with exact counts, plus the dataset's concept names | **Reference only.** Not in the task's `tools`. |

`sketch_id` is deterministic, for example `<dataset>-w<width>-d<depth>`. Rebuilding the same sketch must not be charged twice: propose a rule.

Decide in Checkpoint 3 whether the sketch's estimate per candidate should be returned anywhere besides the saved file, and whether `min_count` and both budgets are locked in the ledger the same way plugin 1 locks threshold and budget.

### Task

- id `find-frequent-pairs`, runner `agent`, tools `[describe_corpus, build_sketch, count_pairs]`, reference `exact_frequent_pairs`.
- Input: `dataset`, `min_count` (integer k), `memory_budget` (total counters across all sketches), `query_budget` (distinct pairs `count_pairs` may count), `min_recall` (decimal string).
- Output: the list of qualifying pairs (concept names a < b).
- Instructions must explain what width and depth do (more width lowers collision error, more depth takes the minimum over more rows), that estimates never fall below true counts, the standard error bound (estimate exceeds the true count by at most e·N/width with probability at least 1 - e^(-depth), N the total pair occurrences), what the histogram means, that a sketch may have more candidates than the query budget, and that an empty list is the correct answer when nothing qualifies. Never name a configuration that works.

### Verifier

Same order as plugin 1, adapted:
1. Malformed: known concepts, a ≠ b, no duplicates in either order.
2. Budgets: ledger matches the task's dataset, min_count and both budgets; memory and query totals within budget; non-empty answer without a ledger fails.
3. Precision: every submitted pair is in the reference.
4. Provenance: `count_pairs` returned every submitted pair as qualifying, with the reference's exact count.
5. Recall: as in plugin 1; score = recall.

`verify_output` rederives each call's meaning from the saved candidate files and the ledger, as plugin 1 does.

### Tests

Mirror plugin 1's coverage: correct; one false pair; empty answer vs non-empty reference; empty reference; duplicates and reversed pairs; unknown concept; self-pair; memory budget exceeded; query budget exceeded; missing ledger; pair never counted; recall exactly at min_recall; a pair with count exactly k and one with k - 1; a keyword repeated within one paper counted once. Tool tests: sketch never underestimates on every pair of `acl-small`; determinism of `build_sketch`; dedup and refusal in `count_pairs`; reference equals a slow independent count.

## Work plan and checkpoints

Stop at each checkpoint, summarize, and wait for my OK.

1. Setup and rename as above. Build `../gapmap-corpus-gen/` and the master corpus. **Checkpoint 1**: paper count per year, papers kept after the 2-concept filter, concept frequency distribution (top 20 and how many concepts never matched), license found.
2. Derive datasets. Measure, for k in a few candidate values (for example 5, 10, 20): qualifying pairs, distinct pairs, and for several (width, depth) settings within a few memory budgets, the candidate count and the recall reached when counting only the top N candidates for a few N. **Checkpoint 2**: show the tables and propose k, memory_budget, query_budget and min_recall per instance so that a careless configuration fails and a sensible one passes. Include one instance whose reference is empty and the edge-cases instance.
3. `contract.py`. **Checkpoint 3**: show it, with the decisions asked above.
4. Tools, manifest, verifier, tests.
5. UI, product page, README (`## Tools`, `## Layout`, `## Develop and run`, with the instance table and the verifier order, like plugin 1).
6. All checks: `uv lock`, `uv sync --extra dev`, `uv run pytest`, `uv run ruff check .`, `uv lock --check`, `nanoagent plugin inspect .`, SKILL.md section 8. Status `ready`. **Checkpoint 4**: results, plus the exact instruction and tool text the model will see.
7. With my OK, create the public repo `concept-cooccurrence-query` and push.

## Report notes

Keep them in `../cooccurrence-report-notes.md`, never in this repo: corpus source, license, matching rule and command; chosen instance parameters with reasons; every verifier case and verdict; test output; limitations (for example: concept sets come from exact phrase matching, so synonyms and inflections are missed; the budgets simulate cost, since the tool itself could count exactly; sketch quality depends on the fixed hash parameters).