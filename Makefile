PY ?= python

.PHONY: help demo test example clean

help:
	@echo "make demo     generate the example table and audit it (~10 seconds)"
	@echo "make test     run the test suite (54 tests, pytest optional)"
	@echo "make example  regenerate examples/example.csv only"
	@echo "make clean    remove generated results"

demo:
	$(PY) demo.py

example:
	$(PY) examples/make_example.py --out examples/example.csv

test:
	$(PY) tests/test_stats.py
	$(PY) tests/test_audit.py

clean:
	rm -rf results/*.json results/*.png
