# leaderboard-audit

Run an evaluation, then find out what its ordering actually supports.

```
python demo.py                  # full pipeline offline: no key, no spend
python tests/test_stats.py      # 24 tests
python tests/test_audit.py      # 26 tests
python tests/test_system.py     # 46 tests
python tests/test_compare.py    # 13 tests
```

`numpy` is the only requirement for everything except a live run. Every
statistical test is implemented here and checked against hand-computed values.

---

## A verified result

Not a simulation. Six real models, 200 real GSM8K items, graded two ways, for
$0.30 of API spend. Every number below came out of this repository.

**Graded deterministically** — take the final number in the answer, compare it
to the reference:

```
6 systems, 2 distinguishable tiers: 4 of the 5 rank gaps are not supported.

system                             score  rank  rank_95%  tier
---------------------------------  -----  ----  --------  ----
meta-llama/llama-3.3-70b-instruct  0.950  1     1-4       1
openai/gpt-4o-mini                 0.945  2     1-4       1
mistralai/ministral-8b-2512        0.935  3     1-4       1
openai/gpt-4.1-nano                0.935  3     1-5       1
qwen/qwen-2.5-7b-instruct          0.895  5     4-5       1
meta-llama/llama-3.1-8b-instruct   0.840  6     6-6       2
```

Five of six models span 89.5% to 95.0% and **none of those five can be told
apart** at 200 items. At this size the table cannot see differences below 5.6
points; separating the closest pair would take about 6,300 items.

**Then the same answers, graded by a model judge, three times each.** The
answers came from cache, so only the grading was paid for:

```
system                             score  rank  ← was
---------------------------------  -----  ----  -----
openai/gpt-4o-mini                 0.943  1     2
qwen/qwen-2.5-7b-instruct          0.936  2     5
mistralai/ministral-8b-2512        0.930  3     3
openai/gpt-4.1-nano                0.915  4     3
meta-llama/llama-3.3-70b-instruct  0.889  5     1
meta-llama/llama-3.1-8b-instruct   0.827  6     6
```

**The first-place model fell to fifth.** Same answers. Only the grader changed.

### Why no amount of repeat grading would have caught this

The judge is not flaky. Graded three times at temperature 0, it gave the same
verdict on **98.3%** of answers, and every one of its 3,592 replies followed
the required output format. Overall it agreed with ground truth **93.1%** of
the time. By every check an evaluation normally runs, this judge is fine.

`la compare` puts the two graders side by side on the same answers:

```
system                             truth  judge  bias    too_generous  too_harsh  holm_p
---------------------------------  -----  -----  ------  ------------  ---------  ------
meta-llama/llama-3.3-70b-instruct  0.950  0.889  -0.060  1             13         0.005
openai/gpt-4.1-nano                0.935  0.915  -0.020  2             6          1.000
meta-llama/llama-3.1-8b-instruct   0.839  0.827  -0.012  9             10         1.000
mistralai/ministral-8b-2512        0.935  0.930  -0.005  6             6          1.000
openai/gpt-4o-mini                 0.945  0.943  -0.002  6             6          1.000
qwen/qwen-2.5-7b-instruct          0.894  0.936  +0.042  13            4          0.245
```

The judge's error is not spread evenly. It is **6.0 points harsh** to one model
and **4.2 points generous** to another — a 10.2-point spread, on a benchmark
where the real gaps between the top five are one to five points. Six pairwise
orderings flip, including first place.

That is bias, not noise, and the distinction is the whole point: **every repeat
is wrong in the same direction**, so repeat grading confirms it instead of
revealing it. Only a second, independent grader on the same answers exposes it.

Reproduce the analysis from the shipped tables:

```bash
la audit   run_gsm8k_numeric/scores.csv
la audit   run_gsm8k_judge/scores.csv
la compare run_gsm8k_numeric/scores.csv run_gsm8k_judge/scores.csv
```

---

## Four commands

```
la estimate --items q.jsonl --systems a,b,c --scorer judge:m   # cost, no calls
la run      --items q.jsonl --systems a,b,c --budget 5         # measure
la audit    run/scores.csv                                     # what holds up
la compare  truth.csv judge.csv                                # grader bias
```

`run` refuses to start without `--budget`. The loop calls a paid API once per
cell, and a loop that works perfectly and bills all night is the failure mode
that matters.

---

## What the audit reports

**Tiers.** How many groups the data can actually separate, against the pairwise
family Holm-corrected. For 31 systems that family is 465 tests; uncorrected,
about 23 come back significant on pure noise.

**Rank intervals.** Items are resampled, keeping every replicate a complete
table, so an item that is hard for everyone stays hard for everyone. A system
whose rank swings from 3rd to 19th does not have a rank, it has a range.

**Where the variance comes from.** Item difficulty inflates raw spread but
cancels under pairing, so it costs nothing — mistaking it for noise is how
people conclude a benchmark is hopeless when it is fine. Repeat-run variance is
what actually sets the floor, and it is invisible without repeat runs.

**What to change.** Separating underpowered pairs, which more items would fix,
from genuinely tied ones, which no item count will.

**The design floor.** Every test reports the smallest p-value it could have
reached. With 5 paired items that floor is 0.0625 — such a comparison cannot
come out significant whatever the systems do.

**Non-transitivity**, reported rather than hidden. "Not distinguishable" is not
transitive, so `tiers` gives the readable partition, `homogeneous_subsets` the
stricter overlapping view, and `nontransitive_pairs` the orderings the
significance pattern cannot support.

---

## What makes it survive a real run

**Everything is cached, keyed by content** — model, messages, temperature, max
tokens and the repeat index. In the run above, all 3,600 gradings reused cached
answers: not one answer was regenerated. A run killed at cell 400 of 600
resumes by replaying 400 hits.

**A published run replays offline.** Ship the cache with the results and
`--offline` re-executes the identical pipeline with no key and no network —
verified here to reproduce the live run cell for cell. A cache miss raises
instead of quietly going online, so a reproduction cannot silently become a
fresh paid run.

**The budget is checked before each call**, priced from the provider's live
model list rather than a hardcoded table that goes stale in the direction
nobody notices until the bill.

**Failures are recorded, not dropped.** Errored cells go to `errors.jsonl` and
are excluded with a count; silently dropping them would bias results toward
whichever systems fail least on hard items — exactly the items that separate
systems. A grade that does not parse is scored 0 and flagged, never discarded.

**Work is ordered item-major**, so an interrupted run still covers every system
on the items it reached. A ragged table cannot be paired.

---

## Input

Items as JSONL or CSV, or `hf:<dataset>` for the Hugging Face hub:

```json
{"id": "q001", "question": "What is 17 * 23?", "answer": "391"}
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

One measurement note found the hard way: the default system prompt tells the
model to answer directly, which suppresses step-by-step reasoning. On GSM8K
that cost about 40 points before `--system-prompt` existed. Set it
deliberately.

---

## What it does not do

It does not say whether a benchmark measures anything worth measuring, or
whether scores transfer to deployment.

It can say whether a grader is *wrong*, but only against a grader you trust
more — that is what `la compare` is. Without a reference it measures
*consistency*, not correctness, and the result above is the reason those are
not the same question.

---

## Layout

```
src/la/
  cli.py           estimate / run / audit / compare
  tasks.py         item loading: JSONL, CSV, Hugging Face
  providers.py     OpenRouter, offline replay, simulator
  judge.py         exact, contains, numeric, model judge, judge panel
  run.py           the measurement loop: resumable, bounded, item-major
  cache.py         content-addressed cache, atomic writes
  budget.py        spend accounting and the hard stop
  env.py           .env loading; names only, never values
  data.py          long-table loading, missing cells, repeat runs
  ranks.py         bootstrap rank intervals, pairwise family, tiers
  variance.py      item / system / repeat-run split, noise floor
  compare.py       per-system grader bias and what it does to the ordering
  report.py        headline, tables, figure, recommendations
  stats/
    exact.py       exact Wilcoxon and sign test, Holm, bootstrap, Cliff's delta
    power.py       minimum detectable effect, required items, achieved power
    normal.py      inverse normal CDF (Acklam), so scipy is not needed
examples/
  gsm8k_test.jsonl     1,319 real items with ground truth
  arithmetic.jsonl     40 quick items
  make_example.py      synthetic table with a known number of true levels
tests/                 109 tests, none touching the network
```

`run/run_summary.json` records config, cell counts, errors, unparsed grades,
cache hit rate and spend. `results/manifest.json` records input shape, alpha,
bootstrap count, seed, items used and dropped, noise floor, and tool, Python,
numpy and platform versions.

## License

Apache-2.0.
