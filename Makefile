.PHONY: help install dev-install format lint type-check test clean run run-study run-sensitivity run-plots figures notebook

help:
	@echo "FMD Sensitivity Analysis - Available Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install        - Install production dependencies"
	@echo "  make dev-install    - Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make format         - Format code with black"
	@echo "  make lint           - Check code with flake8 and ruff"
	@echo "  make type-check     - Run type checking with mypy"
	@echo "  make test           - Run tests with pytest"
	@echo ""
	@echo "Experiments:"
	@echo "  make run-study       - Run the COMPLETE study (all steps, both corpora)"
	@echo "  make run-sensitivity - Run the sensitivity pipeline (steps 3-7, maestro)"
	@echo "  make run-plots       - Regenerate working plots from CSV results"
	@echo "  make figures         - Regenerate paper figures + LaTeX tables from CSVs"
	@echo "  make notebook        - Rebuild and execute final_results.ipynb"
	@echo "  make run             - Alias for run-sensitivity"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          - Clean build artifacts and cache"

install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements.txt

format:
	black src/ tests/ scripts/ main.py

lint:
	flake8 src/ tests/ scripts/ main.py
	ruff check src/ tests/ scripts/ main.py

type-check:
	mypy src/ --ignore-missing-imports

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-quick:
	pytest tests/ -v -k "not integration"

run: run-sensitivity

run-study:
	python scripts/run_full_study.py

run-sensitivity:
	python main.py --mode sensitivity

run-plots:
	python main.py --mode sensitivity --sensitivity-step plots

figures:
	python scripts/generate_draft_figures.py
	python scripts/generate_draft_tables.py

notebook:
	python scripts/build_results_notebook.py
	python -c "import nbformat; from nbconvert.preprocessors import ExecutePreprocessor; nb = nbformat.read('final_results.ipynb', as_version=4); ExecutePreprocessor(timeout=120).preprocess(nb, {'metadata': {'path': '.'}}); nbformat.write(nb, 'final_results.ipynb')"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf build/ dist/ *.egg-info htmlcov/ .pytest_cache/ .coverage
