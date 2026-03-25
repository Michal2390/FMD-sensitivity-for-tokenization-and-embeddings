.PHONY: help install dev-install format lint type-check test clean run run-exp1 run-exp2 run-exp3 run-exp4 run-exp5 run-all docs

help:
	@echo "FMD Sensitivity Analysis - Available Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup and Installation:"
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
	@echo "  make run-exp1       - Run tokenization sensitivity experiment"
	@echo "  make run-exp2       - Run model sensitivity experiment"
	@echo "  make run-exp3       - Run expression ablation experiment"
	@echo "  make run-exp4       - Run quantization sensitivity experiment"
	@echo "  make run-exp5       - Run cross-genre stability experiment"
	@echo "  make run-all        - Run all experiments"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          - Clean build artifacts and cache"
	@echo "  make docs           - Generate documentation"

install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements.txt

format:
	black src/ tests/ run_experiment.py

lint:
	flake8 src/ tests/ run_experiment.py
	ruff check src/ tests/ run_experiment.py

type-check:
	mypy src/ --ignore-missing-imports

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-quick:
	pytest tests/ -v

run-exp1:
	python run_experiment.py --experiment exp1_tokenization_sensitivity

run-exp2:
	python run_experiment.py --experiment exp2_model_sensitivity

run-exp3:
	python run_experiment.py --experiment exp3_expression_ablation

run-exp4:
	python run_experiment.py --experiment exp4_quantization_sensitivity

run-exp5:
	python run_experiment.py --experiment exp5_cross_genre

run-all:
	python run_experiment.py --all

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf build/ dist/ *.egg-info htmlcov/ .pytest_cache/ .coverage

docs:
	@echo "Documentation generation would be implemented here"
	@echo "Using MkDocs or Sphinx for generating HTML docs"

