PY ?= python

.PHONY: help demo test example clean

help:
	@echo "make demo     full pipeline offline: measure 12 systems, then audit"
	@echo "make test     run the test suite (109 tests, pytest optional)"
	@echo "make example  regenerate examples/example.csv only"
	@echo "make clean    remove generated results and run directories"

demo:
	$(PY) demo.py

example:
	$(PY) examples/make_example.py --out examples/example.csv

test:
	$(PY) tests/test_stats.py
	$(PY) tests/test_audit.py
	$(PY) tests/test_system.py
	$(PY) tests/test_compare.py

clean:
	rm -rf results results_live run_demo run_live .cache_demo
