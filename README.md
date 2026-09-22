# leaderboard-audit

Benchmark tables rank systems. They rarely say which of those ranks the data
can actually support. This tool answers one question:

**How many tiers does this leaderboard really have?**

```
make demo     # generates an example table and audits it, ~10 seconds
make test     # 54 tests, pytest optional
```

`numpy` is the only requirement. Every statistical test is implemented here and
checked against hand-computed values, so there is nothing to install before you
see a number.

---

## What it says

`make demo` builds a table of 12 systems drawn from **4 true ability levels**,
three systems per level, identical within a level. The raw means order all 12.
The audit reports this:

```
  12 systems, 4 distinguishable tiers: 8 of the 11 rank gaps are not
  supported by the data.

system    score  rank  rank_95%  span  tier
--------  -----  ----  --------  ----  ----
strong-2  0.786  1     1-3       3     1
strong-3  0.780  2     1-3       3     1
strong-1  0.777  3     1-3       3     1
good-2    0.721  4     4-6       3     2
good-1    0.709  5     4-6       3     2
good-3    0.703  6     4-6       3     2
fair-1    0.642  7     7-9       3     3
fair-3    0.632  8     7-9       3     3
fair-2    0.631  9     7-9       3     3
weak-1    0.446  10    10-12     3     4
weak-2    0.445  11    10-12     3     4
weak-3    0.444  12    10-12     3     4
```

Four tiers is the right answer, and it is the answer because the data says so,
not because the generator was consulted. `strong-2` leads `strong-1` by nine
thousandths of a point and the audit refuses to call that a rank.

The example exists so the tool can be checked against a known truth. Passing
`test_recovers_known_tiers` is the load-bearing claim of this repository.

### Where the spread comes from

With repeat runs present, the variance splits three ways:

```
items       39.9%  (hard for everyone; cancels in pairing)
systems     36.2%  (the part you want to measure)
repeat run  23.9%  (same system, same item, different score)

noise floor: differences below 0.031 are not resolvable at this item count.
```

The distinction matters. Item difficulty inflates the raw spread but cancels
under pairing, so it costs nothing — mistaking it for noise is how people
conclude a benchmark is hopeless when it is fine. Repeat-run variance is the
part that actually sets the floor, and it is invisible without repeat runs.

### What to change

```
- At 60 items this table cannot see differences below about 0.031.
  12 of 66 pairs fall under that line.
- 7 pair(s) are within reach: about 30x more items (60 -> 1770) would
  separate the hardest of them. Adding systems will not help.
- 5 pair(s) differ by less than their own measurement noise. No practical
  item count separates these; report them as tied.
```

The third line matters as much as the first. Some pairs are not underpowered,
they are equal, and quoting an item count for them produces a number in the
millions that nobody can act on.

---

## Input

A long table. Three required columns, one optional:

```csv
system,item,run,score
gpt-x,task_001,1,0.81
gpt-x,task_001,2,0.74
claude-y,task_001,1,0.79
```

`item` is whatever the benchmark has a finite number of — a task, a prompt, a
conversation. `run` is present when the same system was scored on the same item
more than once.

Long format is deliberate. A published leaderboard is wide: one row per system,
one aggregate number. That shape has already discarded everything needed to
say whether the ordering means anything, and **if only the wide table exists,
the audit cannot be run** — which is itself worth reporting.

---

## How it decides

**Items are the resampling unit.** Rank intervals come from resampling items,
keeping every replicate a complete table. An item that is hard for everyone
stays hard for everyone. Resampling scores independently per system would break
the pairing and understate the uncertainty.

**Every comparison is paired.** Systems are evaluated on the same items, so
comparing their marginal means throws away the pairing that makes the
comparison sensitive in the first place.

**The pairwise family is Holm-corrected.** For 31 systems that family is 465
tests; uncorrected, about 23 come back significant on pure noise. This is the
difference between a ranking and a list.

**Tiers are built against the tier leader**, not the previous system. Comparing
only to the neighbour lets a chain of individually-tiny gaps merge into one
tier that spans a real difference end to end.

**Non-transitivity is reported, not hidden.** "Not distinguishable" is not
transitive, so a strict grouping of an ordered table cannot be a clean
partition. `tiers` gives the readable partition; `homogeneous_subsets` gives
the stricter overlapping view; `nontransitive_pairs` names the orderings the
significance pattern cannot support.

**Every test reports the smallest p-value its design could reach.** With 5
paired items that floor is 0.0625 — such a comparison cannot come out
significant no matter what the systems do. That belongs next to the p-value,
not in a footnote.

---

## What it does not do

It does not say whether a benchmark measures anything worth measuring, whether
a grader is *correct*, or whether scores transfer to deployment. It says
whether the numbers in front of it support the ordering placed on them.

Judge correctness needs labelled ground truth. Judge *consistency* does not,
which is why the variance decomposition works from repeat runs alone.

---

## Layout

```
src/la/
  data.py          long-table loading, missing cells, repeat runs
  ranks.py         bootstrap rank intervals, pairwise family, tiers
  variance.py      item / system / repeat-run decomposition, noise floor
  report.py        headline, tables, figure, recommendations
  cli.py           entry point
  stats/
    exact.py       exact Wilcoxon and sign test, Holm, bootstrap, Cliff's delta
    power.py       minimum detectable effect, required items, achieved power
    normal.py      inverse normal CDF (Acklam), so scipy is not needed
examples/
  make_example.py  synthetic table with a known number of true levels
tests/             54 tests
```

`results/manifest.json` records the input shape, alpha, bootstrap count, seed,
items used and dropped, the number of pairwise tests, the noise floor, and the
Python, numpy and platform versions.

## License

Apache-2.0.
