from __future__ import annotations

from trio_core.config import EngineConfig


def test_engine_config_does_not_read_dotenv_implicitly(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "TRIO_REMOTE_VLM_URL=https://example.invalid/v1\nTRIO_REMOTE_VLM_API_KEY=secret\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRIO_REMOTE_VLM_URL", raising=False)
    monkeypatch.delenv("TRIO_REMOTE_VLM_API_KEY", raising=False)

    config = EngineConfig()

    assert config.remote_vlm_url is None
    assert config.remote_vlm_api_key is None


def test_engine_config_from_env_file_reads_explicit_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "TRIO_REMOTE_VLM_URL=https://example.invalid/v1\n"
        "TRIO_REMOTE_VLM_API_KEY=secret\n"
        "TRIO_REMOTE_VLM_MODEL=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TRIO_REMOTE_VLM_URL", raising=False)
    monkeypatch.delenv("TRIO_REMOTE_VLM_API_KEY", raising=False)
    monkeypatch.delenv("TRIO_REMOTE_VLM_MODEL", raising=False)

    config = EngineConfig.from_env_file(env_file)

    assert config.remote_vlm_url == "https://example.invalid/v1"
    assert config.remote_vlm_api_key == "secret"
    assert config.remote_vlm_model == "test-model"


def test_engine_config_explicit_kwargs_override_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "TRIO_REMOTE_VLM_URL=https://example.invalid/v1\nTRIO_REMOTE_VLM_MODEL=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TRIO_REMOTE_VLM_URL", raising=False)
    monkeypatch.delenv("TRIO_REMOTE_VLM_MODEL", raising=False)

    config = EngineConfig.from_env_file(
        env_file,
        remote_vlm_url="https://override.invalid/v1",
        remote_vlm_model="override-model",
    )

    assert config.remote_vlm_url == "https://override.invalid/v1"
    assert config.remote_vlm_model == "override-model"
