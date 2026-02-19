from pathlib import Path

from voxalign.config import load_config


def test_load_dev_profile(monkeypatch) -> None:
    monkeypatch.delenv("VOXALIGN_ENV", raising=False)
    monkeypatch.delenv("VOXALIGN_LOG_LEVEL", raising=False)
    monkeypatch.delenv("VOXALIGN_API_HOST", raising=False)
    monkeypatch.delenv("VOXALIGN_API_PORT", raising=False)
    monkeypatch.delenv("VOXALIGN_WORKERS", raising=False)

    repo_root = Path(__file__).resolve().parents[1]
    config = load_config("dev", config_dir=repo_root / "configs")

    assert config.env == "dev"
    assert config.log_level == "DEBUG"
    assert config.api_host == "127.0.0.1"
    assert config.api_port == 8000
    assert config.workers == 1


def test_env_override(monkeypatch) -> None:
    monkeypatch.setenv("VOXALIGN_ENV", "prod")
    monkeypatch.setenv("VOXALIGN_API_PORT", "9000")

    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(config_dir=repo_root / "configs")

    assert config.env == "prod"
    assert config.api_port == 9000
