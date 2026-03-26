# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.8.x   | :white_check_mark: |
| 0.7.x   | :white_check_mark: |
| < 0.7   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in trio-core, **please report it responsibly**:

1. **Do NOT open a public GitHub issue**
2. Email: **security@machinefi.com**
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Suggested fix (if you have one)
4. You will receive an acknowledgment within **48 hours**
5. We aim to release a fix within **7 days** for critical vulnerabilities
6. We will coordinate disclosure timing with you

## Scope

trio-core processes video/image data and runs ML inference. Key security considerations:

- **API server**: The FastAPI server (`trio serve`) binds to `0.0.0.0` by default. In production, always use a reverse proxy with authentication and TLS.
- **RTSP streams**: Camera credentials are stored in the local database. Protect access to `data/*.db` files.
- **Model loading**: Models are loaded from local ONNX files or HuggingFace Hub. Only use trusted model sources.
- **Cloud APIs**: Gemini API keys for calibration/chat are stored in environment variables. Never commit API keys.
- **Video input**: Video files and RTSP streams are processed via OpenCV. Malformed inputs could potentially trigger OpenCV vulnerabilities.
- **Temporary files**: Inference may create temporary files which are cleaned up after use.
- **Multi-tenant isolation**: API uses tenant-scoped Bearer tokens. Ensure tokens are not shared across tenants.

## Security Best Practices for Deployers

- Run behind a reverse proxy (nginx, Caddy) with TLS
- Use environment variables for all API keys and secrets
- Restrict network access to the API server
- Regularly update trio-core and its dependencies
- Monitor API access logs for anomalous patterns
