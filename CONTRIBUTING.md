# Contributing to trio-core

Thank you for your interest in contributing! This guide will help you get started.

## Quick Start

```bash
# Clone
git clone https://github.com/machinefi/trio-core.git
cd trio-core

# Create virtual environment (requires Python 3.12+)
python -m venv .venv
source .venv/bin/activate

# Install in development mode with uv (recommended)
pip install uv
uv pip install -e '.[dev]'

# Apple Silicon: also install MLX backend
uv pip install -e '.[mlx,dev]'

# NVIDIA GPU: install Transformers backend
uv pip install -e '.[transformers,dev]'
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_counter.py -v

# Run with coverage
python -m pytest tests/ --cov=trio_core
```

## Running the Server

```bash
# Start API server
trio serve

# Start with demo data (no camera or VLM needed)
python -m trio_core.api.demo_server
```

## Code Style

- **Python 3.12+** with type hints everywhere
- Use `from __future__ import annotations` for forward references
- **Linting**: `ruff check .` and `ruff format .`
- Keep functions focused and short
- No unnecessary abstractions -- three similar lines > premature helper
- Comments only where logic isn't self-evident

## License Compliance

**Critical**: Do NOT add any AGPL-licensed dependencies. In particular:
- NO `ultralytics` package -- we use YOLOv10 ONNX + `supervision` (MIT)
- Check the license of any new dependency before adding it
- When in doubt, open an issue to discuss first

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`
2. **Write tests** for any new functionality
3. **Run the test suite** -- all tests must pass
4. **Run linting**: `ruff check . && ruff format --check .`
5. **Run regression tests** if changing inference code:
   ```bash
   python examples/run_regression.py
   ```
6. **Keep PRs focused** -- one feature or fix per PR
7. **Write a clear description** using the PR template

## What to Contribute

### Good First Issues

Look for issues labeled [`good first issue`](https://github.com/machinefi/trio-core/labels/good%20first%20issue) -- these are scoped, actionable tasks designed for new contributors.

### High Impact

- Bug fixes in detection/counting pipeline
- New model support (ONNX profiles + backend testing)
- Performance optimizations with benchmarks
- Counting accuracy improvements (measured against Mall dataset)

### Welcome

- Documentation improvements
- New examples and usage patterns
- CI/CD improvements
- Test coverage for edge cases
- API endpoint additions

### Please Discuss First

- Architecture changes (open an issue)
- New dependencies (license check required)
- API breaking changes

## Project Structure

```
trio-core/
  src/trio_core/
    api/          # FastAPI server, endpoints
    analytics/    # Temporal aggregation, anomaly detection
    insights.py   # InsightExtractor (K3)
    counter.py    # YOLO + ByteTrack counting pipeline
    vlm/          # VLM inference (MLX, Transformers)
  models/         # ONNX model files
  tests/          # Test suite
  experiments/    # Benchmark scripts and results
```

## Reporting Issues

Use the [issue templates](https://github.com/machinefi/trio-core/issues/new/choose) to report bugs, request features, or ask questions. Include:

- Python version, OS, hardware (Apple Silicon model, GPU)
- Model being used
- Minimal reproduction steps
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
