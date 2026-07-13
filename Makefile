UV = uv
PYTHON = $(UV) run python
TEST = $(PYTHON) -m pytest -v -s --no-header
all: install

install:
	$(UV) sync

run: install
	$(PYTHON) -m src $(ARGS)

run-tests: install
	@echo "Running tests..."
	$(TEST) tests/test_tokenizer.py
	$(TEST) tests/test_dataloader.py
	$(TEST) tests/test_attention.py
	$(TEST) tests/test_model.py
	@echo "All tests passed!"


debug:
	$(PYTHON) -m pdb src/__main__.py

clean-all:
	rm -rf __pycache__
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .venv
	rm -rf data/output

clean:
	rm -rf __pycache__
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	$(UV) run flake8 src/
	$(UV) run mypy -p src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 src/
	$(UV) run mypy -p src --strict

.PHONY: all install run run-tests clean-all clean-cache