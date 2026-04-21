.PHONY: help install dev-install format lint type-check test clean run run-multi run-cross-val run-ablation run-interaction run-power run-nfmd run-generative run-shrinkage run-lme run-all paper docs

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
	@echo "  make run-multi      - Run multi-genre 4-model analysis (3840 FMD obs)"
	@echo "  make run-cross-val  - Run cross-dataset validation"
	@echo "  make run-ablation   - Run ablation study"
	@echo "  make run-interaction- Run interaction analysis"
	@echo "  make run-power      - Run sample-size power analysis"
	@echo "  make run-all        - Run all experiments sequentially"
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
	pytest tests/ -v

run-multi:
	python scripts/run_multi_genre_analysis.py

run-cross-val:
	python scripts/run_cross_dataset_validation.py

run-ablation:
	python scripts/run_ablation_study.py

run-interaction:
	python scripts/run_interaction_analysis.py

run-power:
	python scripts/run_sample_size_ablation.py

run-all:
	python scripts/run_multi_genre_analysis.py
	python scripts/run_cross_dataset_validation.py
	python scripts/run_sample_size_ablation.py
	python scripts/run_nfmd_analysis.py
	python scripts/run_generative_eval.py
	python scripts/run_shrinkage_comparison.py
	python scripts/run_lme_analysis.py

run-nfmd:
	python scripts/run_nfmd_analysis.py

run-generative:
	python scripts/run_generative_eval.py

run-shrinkage:
	python scripts/run_shrinkage_comparison.py

run-lme:
	python scripts/run_lme_analysis.py

paper:
	cd paper && pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf build/ dist/ *.egg-info htmlcov/ .pytest_cache/ .coverage

docs:
	@echo "Documentation generation would be implemented here"
	@echo "Using MkDocs or Sphinx for generating HTML docs"

