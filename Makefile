.PHONY: help install dev-install format lint type-check test clean run run-sensitivity run-plots

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
	@echo "  make run-sensitivity - Run full sensitivity pivot (main experiment)"
	@echo "  make run-plots       - Regenerate sensitivity plots from CSV results"
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

run-sensitivity:
	python main.py --mode sensitivity

run-plots:
	python main.py --mode sensitivity --sensitivity-step plots

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf build/ dist/ *.egg-info htmlcov/ .pytest_cache/ .coverage
