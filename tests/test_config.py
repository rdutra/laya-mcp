from __future__ import annotations

import pytest

from laya_mcp.config import DEFAULT_MODEL, Settings


def test_settings_defaults() -> None:
    settings = Settings.from_env({})

    assert settings.model == DEFAULT_MODEL
    assert settings.local_files_only is False
    assert settings.compute_units is None
    assert settings.log_level == "INFO"


def test_settings_parse_environment() -> None:
    settings = Settings.from_env(
        {
            "LAYA_MCP_MODEL": "/models/laya",
            "LAYA_MCP_REVISION": "abc123",
            "LAYA_MCP_LOCAL_FILES_ONLY": "yes",
            "LAYA_MCP_COMPUTE_UNITS": "cpu_ne",
            "LAYA_MCP_LOG_LEVEL": "debug",
        }
    )

    assert settings.model == "/models/laya"
    assert settings.revision == "abc123"
    assert settings.local_files_only is True
    assert settings.compute_units == "cpu_ne"
    assert settings.log_level == "DEBUG"


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"LAYA_MCP_LOCAL_FILES_ONLY": "maybe"}, "LAYA_MCP_LOCAL_FILES_ONLY"),
        ({"LAYA_MCP_COMPUTE_UNITS": "ane"}, "LAYA_MCP_COMPUTE_UNITS"),
        ({"LAYA_MCP_LOG_LEVEL": "verbose"}, "LAYA_MCP_LOG_LEVEL"),
        ({"LAYA_MCP_MODEL": "  "}, "LAYA_MCP_MODEL"),
    ],
)
def test_settings_reject_invalid_values(environment: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Settings.from_env(environment)

