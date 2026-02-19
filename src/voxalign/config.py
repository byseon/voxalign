"""Configuration loading utilities for voxalign."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Application configuration resolved from profile + environment variables."""

    env: str
    log_level: str
    api_host: str
    api_port: int
    workers: int


def load_config(env_name: str | None = None, config_dir: Path | None = None) -> AppConfig:
    """Load configuration from `configs/<env>.toml` and environment overrides."""
    env = env_name or os.getenv("VOXALIGN_ENV", "dev")
    resolved_dir = config_dir or _default_config_dir()
    profile_path = resolved_dir / f"{env}.toml"

    defaults: dict[str, str | int] = {
        "log_level": "INFO",
        "api_host": "127.0.0.1",
        "api_port": 8000,
        "workers": 1,
    }
    file_values = _load_profile(profile_path)
    defaults.update(file_values)

    log_level = os.getenv("VOXALIGN_LOG_LEVEL", str(defaults["log_level"]))
    api_host = os.getenv("VOXALIGN_API_HOST", str(defaults["api_host"]))
    api_port = _parse_int("VOXALIGN_API_PORT", os.getenv("VOXALIGN_API_PORT"), defaults["api_port"])
    workers = _parse_int("VOXALIGN_WORKERS", os.getenv("VOXALIGN_WORKERS"), defaults["workers"])

    return AppConfig(
        env=env,
        log_level=log_level,
        api_host=api_host,
        api_port=api_port,
        workers=workers,
    )


def _default_config_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs"


def _load_profile(path: Path) -> dict[str, str | int]:
    if not path.exists():
        return {}

    with path.open("rb") as handle:
        payload = tomllib.load(handle)

    allowed = {"log_level", "api_host", "api_port", "workers"}
    resolved: dict[str, str | int] = {}
    for key, raw in payload.items():
        if key not in allowed:
            continue
        if key in {"api_port", "workers"}:
            resolved[key] = _coerce_int(key, raw)
        else:
            resolved[key] = _coerce_str(key, raw)
    return resolved


def _parse_int(name: str, raw: str | None, default: str | int) -> int:
    if raw is None:
        return _coerce_int(name, default)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _coerce_int(name: str, value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    raise ValueError(f"{name} must be an integer, got type {type(value).__name__}")


def _coerce_str(name: str, value: object) -> str:
    if isinstance(value, str):
        return value
    raise ValueError(f"{name} must be a string, got type {type(value).__name__}")
