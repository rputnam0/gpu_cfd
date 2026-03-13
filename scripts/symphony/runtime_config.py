#!/usr/bin/env python3
"""Repo-owned runtime configuration for Symphony automation."""

from __future__ import annotations

import pathlib
import shutil
import tomllib
from dataclasses import dataclass


@dataclass(frozen=True)
class CodexProfile:
    model: str
    reasoning_effort: str
    extra_configs: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeConfig:
    codex: dict[str, CodexProfile]


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def default_config_path() -> pathlib.Path:
    return repo_root() / "scripts" / "symphony" / "runtime_config.toml"


def load_runtime_config(path: pathlib.Path | None = None) -> RuntimeConfig:
    config_path = path or default_config_path()
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))

    codex_profiles: dict[str, CodexProfile] = {}
    for name, profile_data in raw.get("codex", {}).items():
        codex_profiles[name] = CodexProfile(
            model=str(profile_data["model"]),
            reasoning_effort=str(profile_data["reasoning_effort"]),
            extra_configs=tuple(
                str(item) for item in profile_data.get("extra_configs", [])
            ),
        )
    return RuntimeConfig(codex=codex_profiles)


def load_codex_profile(name: str) -> CodexProfile:
    config = load_runtime_config()
    if name not in config.codex:
        raise KeyError(f"unknown Codex profile: {name}")
    return config.codex[name]


def resolve_codex_binary() -> str:
    candidates = [
        shutil.which("codex"),
        str(pathlib.Path.home() / ".npm-global" / "bin" / "codex"),
    ]
    for candidate in candidates:
        if candidate and pathlib.Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "could not find codex on PATH or at ~/.npm-global/bin/codex"
    )


def build_codex_command(
    profile_name: str,
    codex_args: list[str],
    *,
    codex_binary: str | None = None,
) -> list[str]:
    profile = load_codex_profile(profile_name)
    command = [codex_binary or resolve_codex_binary(), "-m", profile.model]
    command.extend(["-c", f'model_reasoning_effort="{profile.reasoning_effort}"'])
    for config_override in profile.extra_configs:
        command.extend(["-c", config_override])
    command.extend(codex_args)
    return command
