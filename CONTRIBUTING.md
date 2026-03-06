# Contributing to TrioCore

Thank you for your interest in contributing to TrioCore! This guide will help you get started.

## Development Setup

```bash
# Clone
git clone https://github.com/machinefi/trio-core.git
cd trio-core

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e '.[dev]'

# Apple Silicon: also install MLX backend
pip install -e '.[mlx,dev]'

# NVIDIA GPU: also install Transformers backend
pip install -e '.[transformers,dev]'
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_tome.py -v

# Run with coverage
python -m pytest tests/ --cov=trio_core
```

## Running Benchmarks

```bash
# Quick accuracy check (50 samples, ~2 min)
python examples/run_regression.py --save-baseline

# Full POPE benchmark
python examples/run_benchmark.py --bench pope --samples 100

# Performance eval
python examples/run_eval.py --resolution 1080p --runs 3
```

## Code Style

- Python 3.10+ with type hints
- Use `from __future__ import annotations` for forward references
- Keep functions focused and short
- No unnecessary abstractions — three similar lines > premature helper
- Comments only where logic isn't self-evident

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`
2. **Write tests** for any new functionality
3. **Run the test suite** — all 120+ tests must pass
4. **Run regression tests** if changing inference code:
   ```bash
   python examples/run_regression.py
   ```
5. **Keep PRs focused** — one feature or fix per PR
6. **Write a clear description** of what and why

## What to Contribute

### High Impact
- Bug fixes in inference pipeline
- New model support (profiles + backend testing)
- Performance optimizations with benchmarks
- Accuracy improvements on POPE/TextVQA

### Welcome
- Documentation improvements
- New examples and usage patterns
- CI/CD improvements
- Test coverage for edge cases

### Please Discuss First
- Architecture changes (open an issue)
- New dependencies
- API breaking changes

## Reporting Issues

Use the [issue templates](.github/ISSUE_TEMPLATE/) to report bugs or request features. Include:

- Python version, OS, hardware (M1/M2/M3/M4, GPU model)
- Model being used
- Minimal reproduction steps
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
