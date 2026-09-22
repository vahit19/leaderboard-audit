# leaderboard-audit

Evaluation tables rank systems. They rarely say which of those ranks the data
can support. This runs the evaluation *and* answers that question:

**How many tiers does this leaderboard really have — and how much of the
spread is the grader rather than the systems?**

```
python demo.py                 # full pipeline, offline, no key, no spend
python tests/test_stats.py     # 24 tests
python tests/test_audit.py     # 26 tests
python tests/test_system.py    # 40 tests
```

`numpy` is the only requirement for everything except a live run. Every
statistical test is implemented here and checked against hand-computed values.

---

## Three commands

```
la estimate --items q.jsonl --systems a,b,c --scorer judge:m   # cost, no calls
la run      --items q.jsonl --systems a,b,c --budget 5         # measure
la audit    run/scores.csv                                     # analyse
```

`run` refuses to start without `--budget`. The loop calls a paid API once per
cell, and a loop that works perfectly and bills all night is the failure mode
that matters.

You can also skip straight to `audit` with a table you already have.

---

## What the demo shows

`python demo.py` measures twelve simulated systems on 40 real items with a
model judge, grading **every answer three times**, then audits the result.
The twelve come from four true ability levels — 0.86, 0.74, 0.61, 0.40 — three
ids each. The audit is not told this.

```
  12 systems, 3 distinguishable tiers: 9 of the 11 rank gaps are not
  supported by the data.

system         score  rank  rank_95%  span  tier
-------------  -----  ----  --------  ----  ----
stub/strong-2  0.875  1     1-3       3     1
stub/strong-3  0.875  2     1-3       3     1
stub/strong-1  0.850  3     1-3       3     1
stub/good-1    0.717  4     4-8       5     1
stub/good-2    0.717  4     4-8       5     1
stub/good-3    0.717  6     4-8       5     1
stub/fair-1    0.675  7     4-9       6     2
stub/fair-3    0.625  8     5-9       5     2
stub/fair-2    0.608  9     6-9       4     2
stub/weak-1    0.475  10    9-11      3     3
stub/weak-2    0.433  11    10-12     3     3
stub/weak-3    0.342  12    11-12     2     3
```

Three tiers, not four. The audit merged `strong` and `good` — and it is right
to. A binary judge over 40 items cannot separate abilities twelve points apart,
and the next section says exactly why:

```
items        0.0%  (hard for everyone; cancels in pairing)
systems     12.5%  (the part you want to measure)
repeat run  87.5%  (same system, same item, different score)

repeat-run noise exceeds the spread between systems (87% vs 13%).
noise floor: differences below 0.161 are not resolvable at this item count.
```

**87% of the variance in that table is the grader disagreeing with itself.**
Grading once would have produced the same twelve-way ranking with no hint that
this was true. That number is the reason this repository runs the evaluation
instead of only reading its output — repeat grading is the only way to get it,
and almost nothing repeats the grading.

Then, what to change:

```
- At 40 items this table cannot see differences below about 0.168.
  34 of 66 pairs fall under that line.
- 29 pair(s) are within reach: about 29x more items (40 -> 1172) would
  separate the hardest of them. Adding systems will not help.
- 5 pair(s) differ by less than their own measurement noise. No practical
  item count separates these; report them as tied.
- Repeat-run noise is 87% of total variance. Averaging over more runs per
  item buys resolution more cheaply than adding items.
```

The last line is the actionable one, and it is the opposite of what a team
usually does when a leaderboard looks noisy.

---

## Running it live

```
python demo.py --live --budget 2
```

Estimates first, prints the projection, asks, and only then spends — under a
ceiling it cannot cross. Needs `OPENROUTER_API_KEY` in the environment or a
`.env` file at the repository root. Keys are read into the process environment
and never written to results, the cache, the manifest or a log line.

Directly:

```bash
la run --items questions.jsonl \
       --systems openai/gpt-4o-mini,anthropic/claude-3.5-haiku \
       --panel openai/gpt-4o-mini,google/gemini-flash-1.5 \
       --grade-repeats 3 --budget 5
la audit run/scores.csv --top 10
```

`--panel` grades with several judges and keeps every individual grade, so
disagreement *between* graders is measurable alongside disagreement *within*
one.

---

## What makes it survive a real run

**Everything is cached, keyed by content.** Model, messages, temperature, max
tokens and the repeat index. A run killed at cell 400 of 600 resumes by
replaying 400 cache hits; re-analysis costs nothing. The repeat index is part
of the key on purpose — repeat runs are supposed to differ, so sample 2 must
not be served sample 1's answer.

**A published run can be replayed offline.** Ship the cache with the results
and `--offline` re-executes the identical pipeline with no key and no network.
A cache miss raises instead of quietly going online, so a "reproduction" cannot
silently become a fresh paid run.

**The budget is checked before each call, not after**, using live prices from
the provider rather than a hardcoded table that goes stale in the direction
nobody notices until the bill.

**Failures are recorded, not dropped.** A cell that errors is written to
`errors.jsonl` and excluded, and the count is reported. Silently dropping
failed cells biases results toward whichever systems fail least on hard items —
exactly the items that separate systems.

**A grade that does not parse is scored 0 and flagged**, never discarded. A
judge that ignores its own output format on 8% of items is a finding.

**Work is ordered item-major**, so an interrupted run still covers every system
on the items it reached. System-major order would leave a ragged table, and a
ragged table cannot be paired.

---

## How the audit decides

**Items are the resampling unit.** Rank intervals resample items, keeping every
replicate a complete table, so an item that is hard for everyone stays hard for
everyone. Resampling per system would break the pairing and understate the
uncertainty.

**The pairwise family is Holm-corrected.** For 31 systems that family is 465
tests; uncorrected, about 23 come back significant on pure noise.

**Tiers are built against the tier leader**, not the previous system, so a
chain of individually-tiny gaps cannot merge into one tier spanning a real
difference end to end.

**Non-transitivity is reported, not hidden.** "Not distinguishable" is not
transitive. `tiers` gives the readable partition, `homogeneous_subsets` the
stricter overlapping view, `nontransitive_pairs` the orderings the significance
pattern cannot support.

**Every test reports the smallest p-value its design could reach.** With 5
paired items that floor is 0.0625 — such a comparison cannot come out
significant whatever the systems do.

---

## Input

Items, as JSONL or CSV, or `hf:<dataset>` for the Hugging Face hub:

```json
{"id": "q001", "question": "What is 17 * 23?", "answer": "391", "topic": "*"}
```

Field names are given with `--prompt-field` / `--reference-field` rather than
guessed, because guessing column names is how a harness silently evaluates the
wrong field.

Or audit a table you already have:

```csv
system,item,run,score
gpt-x,task_001,1,0.81
gpt-x,task_001,2,0.74
```

A published leaderboard is wide — one row per system, one aggregate number —
and that shape has already discarded everything needed to say whether the
ordering means anything. **If only the wide table exists, the audit cannot be
run**, which is itself worth reporting.

---

## What it does not do

It does not say whether a benchmark measures anything worth measuring, whether
a grader is *correct*, or whether scores transfer to deployment.

Judge correctness needs labelled ground truth. Judge *consistency* does not,
which is why the variance decomposition works from repeat grading alone — and
consistency is the part nobody measures.

---

## Layout

```
src/la/
  cli.py           estimate / run / audit
  tasks.py         item loading: JSONL, CSV, Hugging Face
  providers.py     OpenRouter, offline replay, simulator
  judge.py         exact match, contains, model judge, judge panel
  run.py           the measurement loop: resumable, bounded, item-major
  cache.py         content-addressed cache, atomic writes
  budget.py        spend accounting and the hard stop
  env.py           .env loading; names only, never values
  data.py          long-table loading, missing cells, repeat runs
  ranks.py         bootstrap rank intervals, pairwise family, tiers
  variance.py      item / system / repeat-run split, noise floor
  report.py        headline, tables, figure, recommendations
  stats/
    exact.py       exact Wilcoxon and sign test, Holm, bootstrap, Cliff's delta
    power.py       minimum detectable effect, required items, achieved power
    normal.py      inverse normal CDF (Acklam), so scipy is not needed
examples/
  arithmetic.jsonl     40 items with checkable answers
  make_example.py      synthetic table with a known number of true levels
tests/                 90 tests
```

`run/run_summary.json` records the config, cell counts, errors, unparsed
grades, cache hit rate and spend. `results/manifest.json` records the input
shape, alpha, bootstrap count, seed, items used and dropped, the noise floor,
and tool, Python, numpy and platform versions.

## License

Apache-2.0.
