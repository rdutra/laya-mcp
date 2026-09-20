"""Environment-backed server configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping

DEFAULT_MODEL = "aac6fef/laya-multilingual-coreml-ane"
_COMPUTE_UNITS = frozenset({"all", "cpu", "cpu_gpu", "cpu_ne"})
_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def _parse_bool(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of true/false, yes/no, on/off, or 1/0")


@dataclass(frozen=True, slots=True)
class Settings:
    model: str = DEFAULT_MODEL
    revision: str | None = None
    local_files_only: bool = False
    compute_units: str | None = None
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ
        model = env.get("LAYA_MCP_MODEL", DEFAULT_MODEL).strip()
        if not model:
            raise ValueError("LAYA_MCP_MODEL must not be empty")

        revision = env.get("LAYA_MCP_REVISION") or None
        local_only = _parse_bool(
            "LAYA_MCP_LOCAL_FILES_ONLY", env.get("LAYA_MCP_LOCAL_FILES_ONLY", "false")
        )
        compute_units = env.get("LAYA_MCP_COMPUTE_UNITS") or None
        if compute_units not in _COMPUTE_UNITS | {None}:
            allowed = ", ".join(sorted(_COMPUTE_UNITS))
            raise ValueError(f"LAYA_MCP_COMPUTE_UNITS must be one of: {allowed}")

        log_level = env.get("LAYA_MCP_LOG_LEVEL", "INFO").upper()
        if log_level not in _LOG_LEVELS:
            raise ValueError(f"LAYA_MCP_LOG_LEVEL must be one of: {', '.join(sorted(_LOG_LEVELS))}")

        return cls(
            model=model,
            revision=revision,
            local_files_only=local_only,
            compute_units=compute_units,
            log_level=log_level,
        )

