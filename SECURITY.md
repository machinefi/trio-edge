# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in TrioCore, please report it responsibly:

1. **Do not** open a public issue
2. Email the maintainers at the address listed in `pyproject.toml`
3. Include a description of the vulnerability and steps to reproduce
4. Allow reasonable time for a fix before public disclosure

## Scope

TrioCore processes video/image data and runs ML inference locally. Key security considerations:

- **Model loading**: Models are loaded from HuggingFace Hub or local paths. Only use trusted model sources.
- **API server**: The FastAPI server (`trio-core serve`) binds to `0.0.0.0` by default. In production, use a reverse proxy and authentication.
- **Video input**: Video files and streams are processed via OpenCV. Malformed inputs could potentially trigger OpenCV vulnerabilities.
- **Temporary files**: The MLX backend creates temporary video files during inference. These are cleaned up after use.
