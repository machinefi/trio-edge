---
name: Bug Report
about: Report a bug in TrioCore
title: "[Bug] "
labels: bug
assignees: ''
---

## Description

A clear description of the bug.

## Environment

- **OS**: macOS / Linux / Windows
- **Hardware**: Apple M3 Pro 18GB / RTX 4090 / CPU only
- **Python**: 3.12
- **trio-core version**: 0.2.1
- **Backend**: mlx / transformers

## Model

- **Model ID**: `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`

## Steps to Reproduce

```python
from trio_core import TrioCore
engine = TrioCore()
engine.load()
# ...
```

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happened. Include error messages and tracebacks.

## Additional Context

- [ ] ToMe enabled? If so, what `r` value?
- [ ] Streaming or batch inference?
- [ ] Happens with all models or specific ones?
