"""Tests for the measurement system: cache, budget, providers, run, scoring.

These cover the properties that make the pipeline usable on paid work rather
than on a demo. A resume that silently re-pays, or a budget ceiling that is
checked after the call instead of before, would not show up in any analysis
test and would cost real money.

Nothing here touches the network.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from la.budget import Budget, BudgetExceeded, Price, estimate_tokens  # noqa: E402
from la.cache import CallKey, DiskCache  # noqa: E402
from la.env import load_dotenv, loaded_keys  # noqa: E402
from la.judge import (Contains, ExactMatch, JudgePanel, ModelJudge,  # noqa: E402
                      NumericMatch, build_scorer)
from la.providers import (Completion, OpenRouterProvider,  # noqa: E402
                          ProviderError, ReplayProvider, StubProvider)
from la.run import (RunConfig, load_results, run_evaluation,  # noqa: E402
                    write_long_csv)
from la.tasks import Item, load_jsonl, slice_by  # noqa: E402


class temp_dir:
    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="la-test-")
        return self.path

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)


MESSAGES = [{"role": "user", "content": "2+2?"}]


def items(n=6):
    return [Item("q%d" % i, "question %d" % i, "answer %d" % i,
                 {"topic": "a" if i % 2 else "b"})
            for i in range(n)]


# -- cache ------------------------------------------------------------------

def test_cache_round_trips_a_completion():
    with temp_dir() as root:
        cache = DiskCache(root)
        key = CallKey("m", MESSAGES, 0.0, 16)
        assert cache.get(key) is None
        cache.put(key, {"text": "4"})
        assert cache.get(key)["text"] == "4"
        assert cache.stats()["hits"] == 1


def test_cache_key_separates_repeat_samples():
    """Sample 2 must not be served sample 1's answer, or repeat runs would be
    identical by construction and measure nothing."""
    a = CallKey("m", MESSAGES, 0.7, 16, sample=1)
    b = CallKey("m", MESSAGES, 0.7, 16, sample=2)
    assert a.digest() != b.digest()


def test_cache_key_covers_every_parameter_that_changes_the_answer():
    base = CallKey("m", MESSAGES, 0.0, 16)
    variants = [
        CallKey("other", MESSAGES, 0.0, 16),
        CallKey("m", [{"role": "user", "content": "3+3?"}], 0.0, 16),
        CallKey("m", MESSAGES, 0.5, 16),
        CallKey("m", MESSAGES, 0.0, 32),
    ]
    for v in variants:
        assert v.digest() != base.digest()


def test_cache_survives_a_truncated_file():
    with temp_dir() as root:
        cache = DiskCache(root)
        key = CallKey("m", MESSAGES, 0.0, 16)
        cache.put(key, {"text": "4"})
        path = cache._path(key.digest())
        with open(path, "w", encoding="utf-8") as fh:
            fh.write('{"text": "4"')          # killed mid-write
        assert cache.get(key) is None         # a miss, not a crash


def test_disabled_cache_stores_nothing():
    with temp_dir() as root:
        cache = DiskCache(root, enabled=False)
        key = CallKey("m", MESSAGES, 0.0, 16)
        cache.put(key, {"text": "4"})
        assert cache.get(key) is None


# -- budget -----------------------------------------------------------------

def test_budget_blocks_before_the_call_not_after():
    budget = Budget(limit_usd=1.0)
    budget.record(1_000_000, 0, Price(0.9, 0.0))
    try:
        budget.check(0.5)
    except BudgetExceeded:
        return
    raise AssertionError("a call crossing the ceiling should be refused")


def test_budget_allows_a_call_that_fits():
    budget = Budget(limit_usd=1.0)
    budget.record(100_000, 0, Price(0.5, 0.0))     # $0.05
    budget.check(0.10)                             # still under


def test_budget_without_a_limit_never_blocks():
    budget = Budget(None)
    budget.record(10_000_000, 10_000_000, Price(50.0, 50.0))
    budget.check(1000.0)
    assert budget.remaining() is None


def test_price_arithmetic_is_per_million_tokens():
    price = Price(prompt=3.0, completion=15.0)
    assert abs(price.cost(1_000_000, 0) - 3.0) < 1e-12
    assert abs(price.cost(0, 1_000_000) - 15.0) < 1e-12
    assert abs(price.cost(500_000, 100_000) - (1.5 + 1.5)) < 1e-12


def test_cached_calls_cost_nothing_but_are_counted():
    budget = Budget(limit_usd=1.0)
    assert budget.record(999, 999, Price(100, 100), cached=True) == 0.0
    assert budget.summary()["cached_calls"] == 1
    assert budget.summary()["spent_usd"] == 0.0


def test_reported_cost_wins_over_our_own_multiplication():
    budget = Budget()
    budget.record(1000, 1000, Price(1.0, 1.0), cost_usd=0.5)
    assert abs(budget.summary()["spent_usd"] - 0.5) < 1e-12


def test_estimate_tokens_is_monotone_and_never_zero():
    assert estimate_tokens("") >= 1
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)


# -- providers --------------------------------------------------------------

def test_replay_serves_the_cache_and_refuses_to_go_online():
    with temp_dir() as root:
        cache = DiskCache(root)
        key = CallKey("m", MESSAGES, 0.0, 16, sample=1)
        cache.put(key, Completion("4", "m", 3, 1, 0.01).to_dict())

        provider = ReplayProvider(cache)
        got = provider.complete(MESSAGES, "m", 0.0, 16, sample=1)
        assert got.text == "4" and got.cached

        try:
            provider.complete(MESSAGES, "m", 0.0, 16, sample=2)
        except ProviderError as exc:
            assert "replay miss" in str(exc)
            return
        raise AssertionError("a replay miss must raise, not fetch")


def test_openrouter_refuses_to_construct_without_a_key():
    saved = {k: os.environ.pop(k) for k in
             ("OPENROUTER_API_KEY", "openrouter_api_key") if k in os.environ}
    try:
        OpenRouterProvider()
    except ProviderError as exc:
        assert "OPENROUTER_API_KEY" in str(exc)
    else:
        raise AssertionError("missing key should raise")
    finally:
        os.environ.update(saved)


def test_stub_gives_each_model_a_stable_accuracy():
    stub = StubProvider(accuracies={"a": 0.9, "b": 0.3})
    assert stub.true_accuracy("a") == 0.9
    assert stub.true_accuracy("b") == 0.3
    # Unlisted ids get a derived value that is stable across instances.
    assert StubProvider().true_accuracy("x") == StubProvider().true_accuracy("x")


def test_stub_answers_differ_by_model_and_by_sample():
    stub = StubProvider()
    a1 = stub.complete(MESSAGES, "m1", sample=1).text
    a2 = stub.complete(MESSAGES, "m2", sample=1).text
    a3 = stub.complete(MESSAGES, "m1", sample=2).text
    assert a1 != a2 and a1 != a3


def test_stub_judge_tracks_the_answering_model_accuracy():
    stub = StubProvider(accuracies={"hi": 0.95, "lo": 0.05}, judge_noise=0.0)
    judge = ModelJudge(stub, "judge-model")

    def rate(model):
        hits = 0
        for i in range(120):
            answer = stub.complete([{"role": "user", "content": "q%d" % i}],
                                   model).text
            hits += judge.score("q%d" % i, answer, "ref").value
        return hits / 120.0

    assert rate("hi") > 0.8
    assert rate("lo") < 0.2


def test_stub_judge_noise_makes_repeat_gradings_disagree():
    stub = StubProvider(accuracies={"m": 0.5}, judge_noise=0.4)
    judge = ModelJudge(stub, "j")
    answer = stub.complete(MESSAGES, "m").text
    grades = {judge.score("q", answer, "ref", sample=s).value
              for s in range(1, 8)}
    assert len(grades) > 1, "a noisy judge must not be perfectly repeatable"


# -- scorers ----------------------------------------------------------------

def test_exact_match_normalises_case_and_punctuation():
    scorer = ExactMatch()
    assert scorer.score("q", "  Paris. ", "paris").value == 1.0
    assert scorer.score("q", "London", "paris").value == 0.0


def test_contains_finds_the_reference_inside_a_sentence():
    scorer = Contains()
    assert scorer.score("q", "I think the answer is 42.", "42").value == 1.0
    assert scorer.score("q", "I think it is 41.", "42").value == 0.0


def test_numeric_match_reads_the_last_number_after_reasoning():
    """Models that reason aloud state the result last; the scorer must take
    that one, not the first number it sees in the working."""
    scorer = NumericMatch()
    reasoning = ("She starts with 16 eggs, eats 3, bakes with 4, so 16-3-4=9 "
                 "left. At $2 each that is 18.")
    assert scorer.score("q", reasoning, "18").value == 1.0
    assert scorer.score("q", reasoning, "9").value == 0.0


def test_numeric_match_strips_formatting_models_actually_produce():
    scorer = NumericMatch()
    for answer in ("The answer is 1,250.", "$1250", "1250.0", "**1250**"):
        assert scorer.score("q", answer, "1250").value == 1.0, answer


def test_numeric_match_handles_latex_style_output():
    scorer = NumericMatch()
    latex = r"\[ 	ext{Total} = 2 + 1 = 3 	ext{ bolts} \] Thus: 3"
    assert scorer.score("q", latex, "3").value == 1.0


def test_numeric_match_flags_an_answer_with_no_number():
    scorer = NumericMatch()
    result = scorer.score("q", "I cannot determine this.", "42")
    assert result.value == 0.0
    assert result.parsed is False


def test_numeric_match_rejects_a_reference_without_a_number():
    try:
        NumericMatch().score("q", "5", "five")
    except ValueError as exc:
        assert "no number" in str(exc)
        return
    raise AssertionError("a non-numeric reference should raise")


def test_numeric_match_is_deterministic_across_samples():
    scorer = NumericMatch()
    values = {scorer.score("q", "the answer is 7", "7", sample=s).value
              for s in range(5)}
    assert values == {1.0}


def test_deterministic_scorers_need_a_reference():
    for scorer in (ExactMatch(), Contains(), NumericMatch()):
        try:
            scorer.score("q", "a", None)
        except ValueError:
            continue
        raise AssertionError("%s should require a reference" % scorer.name)


def test_model_judge_records_an_unparseable_reply_instead_of_dropping_it():
    class Rambling(StubProvider):
        def complete(self, messages, model, temperature=0.0, max_tokens=1024,
                     sample=0):
            return Completion("I'd rather not say.", model)

    score = ModelJudge(Rambling(), "j").score("q", "a", "ref")
    assert score.parsed is False
    assert score.value == 0.0
    assert score.raw


def test_model_judge_parses_the_required_format():
    class Grader(StubProvider):
        def __init__(self, verdict):
            super().__init__()
            self.verdict = verdict

        def complete(self, messages, model, temperature=0.0, max_tokens=1024,
                     sample=0):
            return Completion("GRADE: %s" % self.verdict, model)

    assert ModelJudge(Grader("CORRECT"), "j").score("q", "a", "r").value == 1.0
    assert ModelJudge(Grader("INCORRECT"), "j").score("q", "a", "r").value == 0.0


def test_panel_averages_and_keeps_every_grade():
    class Fixed:
        def __init__(self, value):
            self.value = value
            self.name = "fixed%s" % value

        def score(self, prompt, answer, reference, sample=0):
            from la.judge import Score
            return Score(self.value, self.name)

    panel = JudgePanel([Fixed(1.0), Fixed(1.0), Fixed(0.0)])
    result = panel.score("q", "a", "r")
    assert abs(result.value - 2.0 / 3.0) < 1e-12
    assert len(panel.last_scores) == 3
    assert abs(panel.agreement() - 2.0 / 3.0) < 1e-12


def test_build_scorer_rejects_an_unknown_spec():
    try:
        build_scorer("magic")
    except ValueError as exc:
        assert "unknown scorer" in str(exc)
        return
    raise AssertionError("unknown scorer should raise")


# -- the run ----------------------------------------------------------------

def _config(out_dir, n_items=4, grade_repeats=1, systems=("a", "b")):
    return RunConfig(systems=list(systems), items=items(n_items),
                     scorer=ExactMatch(), grade_repeats=grade_repeats,
                     concurrency=2, out_dir=out_dir)


def test_run_writes_one_row_per_cell():
    with temp_dir() as out:
        config = _config(out, n_items=4, grade_repeats=2)
        result = run_evaluation(config, StubProvider(), Budget())
        assert len(result.rows) == config.total_cells == 2 * 4 * 1 * 2


def test_run_resumes_without_repeating_finished_cells():
    with temp_dir() as out:
        config = _config(out, n_items=5)
        first = StubProvider()
        run_evaluation(config, first, Budget())
        calls_first = first.calls

        second = StubProvider()
        again = run_evaluation(config, second, Budget())
        assert second.calls == 0, "a resumed run must not recall the model"
        assert len(again.rows) == config.total_cells
        assert calls_first > 0


def test_no_resume_starts_over_and_keeps_a_backup():
    with temp_dir() as out:
        config = _config(out, n_items=3)
        run_evaluation(config, StubProvider(), Budget())
        provider = StubProvider()
        run_evaluation(config, provider, Budget(), resume=False)
        assert provider.calls > 0
        assert os.path.exists(os.path.join(out, "results.jsonl.bak"))


def test_results_file_is_valid_jsonl_after_a_run():
    with temp_dir() as out:
        config = _config(out, n_items=3)
        run_evaluation(config, StubProvider(), Budget())
        path = os.path.join(out, "results.jsonl")
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        assert len(rows) == config.total_cells
        assert load_results(path).keys() == {r["cell"] for r in rows}


def test_run_stops_when_the_budget_is_exhausted():
    with temp_dir() as out:
        class Expensive(StubProvider):
            def complete(self, messages, model, temperature=0.0,
                         max_tokens=1024, sample=0):
                self.budget.check(10.0)
                self.budget.record(0, 0, cost_usd=10.0)
                return Completion("x", model)

        provider = Expensive()
        provider.budget = Budget(limit_usd=25.0)
        config = _config(out, n_items=20)
        result = run_evaluation(config, provider, provider.budget)
        assert result.stopped_early
        assert "limit" in result.stop_reason
        assert len(result.rows) < config.total_cells


def test_a_failing_cell_is_recorded_and_does_not_stop_the_run():
    with temp_dir() as out:
        class Flaky(StubProvider):
            def complete(self, messages, model, temperature=0.0,
                         max_tokens=1024, sample=0):
                if "question 2" in "".join(m["content"] for m in messages):
                    raise ProviderError("upstream exploded")
                return super().complete(messages, model, temperature,
                                        max_tokens, sample)

        config = _config(out, n_items=5)
        result = run_evaluation(config, Flaky(), Budget())
        assert len(result.errors) == 2               # one per system
        assert len(result.rows) == config.total_cells - 2
        assert os.path.exists(os.path.join(out, "errors.jsonl"))


def test_run_refuses_repeat_answers_at_temperature_zero():
    with temp_dir() as out:
        try:
            RunConfig(systems=["a", "b"], items=items(3), scorer=ExactMatch(),
                      answer_repeats=3, temperature=0.0, out_dir=out)
        except ValueError as exc:
            assert "temperature" in str(exc)
            return
        raise AssertionError("repeat answers at T=0 measure nothing")


def test_work_is_ordered_so_an_interrupted_run_stays_paired():
    """Every system must be covered on an item before moving on, otherwise a
    stopped run leaves a ragged table that cannot be paired."""
    from la.run import _plan
    with temp_dir() as out:
        config = _config(out, n_items=4, systems=("a", "b", "c"))
        plan = _plan(config, {})
        first_three = [unit[0] for unit in plan[:3]]
        assert sorted(first_three) == ["a", "b", "c"]


# -- handoff to the audit ---------------------------------------------------

def test_long_csv_keeps_each_repeat_as_its_own_row():
    with temp_dir() as out:
        config = _config(out, n_items=3, grade_repeats=3)
        result = run_evaluation(config, StubProvider(), Budget())
        path = os.path.join(out, "scores.csv")
        written, skipped = write_long_csv(result.rows, path)
        assert written == config.total_cells and skipped == 0

        from la.data import load_csv
        table = load_csv(path)
        assert table.has_repeats
        assert table.n_systems == 2 and table.n_items == 3


def test_long_csv_can_drop_unparsed_grades_and_reports_how_many():
    rows = [
        {"system": "a", "item": "x", "score": 1.0, "parsed": True},
        {"system": "a", "item": "y", "score": 0.0, "parsed": False},
    ]
    with temp_dir() as out:
        path = os.path.join(out, "scores.csv")
        written, skipped = write_long_csv(rows, path, drop_unparsed=True)
        assert (written, skipped) == (1, 1)


# -- tasks and env ----------------------------------------------------------

def test_jsonl_loader_reports_a_missing_field_with_the_available_ones():
    with temp_dir() as out:
        path = os.path.join(out, "d.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"q": "x", "a": "y"}) + "\n")
        try:
            load_jsonl(path, prompt_field="question")
        except ValueError as exc:
            assert "question" in str(exc) and "q" in str(exc)
            return
        raise AssertionError("missing field should raise")


def test_duplicate_item_ids_are_refused():
    with temp_dir() as out:
        path = os.path.join(out, "d.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for _ in range(2):
                fh.write(json.dumps({"id": "same", "question": "q",
                                     "answer": "a"}) + "\n")
        try:
            load_jsonl(path, id_field="id")
        except ValueError as exc:
            assert "duplicate" in str(exc)
            return
        raise AssertionError("duplicate ids would merge two questions")


def test_slice_by_groups_items_on_metadata():
    grouped = slice_by(items(6), "topic")
    assert set(grouped) == {"a", "b"}
    assert sum(len(v) for v in grouped.values()) == 6


def test_dotenv_loads_names_without_exposing_values():
    with temp_dir() as out:
        path = os.path.join(out, ".env")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write('LA_TEST_KEY="secret-value"\n# comment\n\n')
        names = load_dotenv(path, override=True)
        assert "LA_TEST_KEY" in names
        assert os.environ["LA_TEST_KEY"] == "secret-value"
        # loaded_keys reports presence only, never a value
        assert set(loaded_keys().values()) <= {True, False}
        del os.environ["LA_TEST_KEY"]


def test_dotenv_does_not_override_the_real_environment_by_default():
    os.environ["LA_TEST_EXISTING"] = "from-environment"
    with temp_dir() as out:
        path = os.path.join(out, ".env")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("LA_TEST_EXISTING=from-file\n")
        load_dotenv(path)
        assert os.environ["LA_TEST_EXISTING"] == "from-environment"
    del os.environ["LA_TEST_EXISTING"]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("PASS  %s" % name)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("FAIL  %s: %s" % (name, exc))
    print("\n%d failed" % failures if failures else "\nall passed")
    raise SystemExit(1 if failures else 0)
