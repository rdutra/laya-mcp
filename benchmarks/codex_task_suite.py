"""Small reproducible coding-task fixtures for the Codex A/B experiment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CodingTask:
    task_id: str
    title: str
    prompt: str
    files: dict[str, str]
    validation: tuple[str, ...] = ("python", "-m", "pytest", "-q")
    must_change: tuple[str, ...] = ()


def _project(*files: tuple[str, str]) -> dict[str, str]:
    base = {
        "pyproject.toml": """[project]
name = "mini-coding-fixture"
version = "0.1.0"
requires-python = ">=3.12"

[tool.pytest.ini_options]
pythonpath = ["."]
""",
        "app/__init__.py": "",
    }
    base.update(dict(files))
    return base


def _large_candidate_project() -> dict[str, str]:
    files = _project(
        (
            "app/cart.py",
            """def total_cents(items: list[tuple[int, int]]) -> int:
    # quantity is accidentally ignored for multi-item carts.
    return sum(price for price, _quantity in items)
""",
        ),
        (
            "tests/test_cart.py",
            """from app.cart import total_cents


def test_cart_total_uses_quantity() -> None:
    assert total_cents([(250, 2), (100, 1)]) == 600
""",
        ),
    )
    distractors = {
        "auth": "def check_password(value: str) -> bool:\n    return len(value) >= 12\n",
        "audio": "def normalize_gain(db: float) -> float:\n    return max(-60.0, min(6.0, db))\n",
        "cache": "def cache_key(user_id: str) -> str:\n    return f'user:{user_id}'\n",
        "cli": "def format_help(command: str) -> str:\n    return f'Usage: {command}'\n",
        "config": "def default_port() -> int:\n    return 8080\n",
        "crypto": "def constant_time_equal(left: bytes, right: bytes) -> bool:\n    return left == right\n",
        "email": "def normalize_address(value: str) -> str:\n    return value.strip().lower()\n",
        "inventory": "def available(stock: int, reserved: int) -> int:\n    return max(0, stock - reserved)\n",
        "logging": "def event_name(value: str) -> str:\n    return value.replace(' ', '_').lower()\n",
        "metrics": "def ratio(done: int, total: int) -> float:\n    return done / total if total else 0.0\n",
        "notifications": "def subject(value: str) -> str:\n    return value[:80]\n",
        "permissions": "def can_read(role: str) -> bool:\n    return role in {'admin', 'reader'}\n",
        "search": "def normalize_query(value: str) -> str:\n    return ' '.join(value.split())\n",
        "shipping": "def zone(country: str) -> str:\n    return 'domestic' if country == 'UY' else 'international'\n",
        "telemetry": "def sample(rate: float) -> bool:\n    return rate >= 1.0\n",
        "templates": "def render_name(name: str) -> str:\n    return name.strip().title()\n",
        "users": "def display_name(first: str, last: str) -> str:\n    return f'{first} {last}'.strip()\n",
        "webhooks": "def event_type(payload: dict[str, str]) -> str:\n    return payload.get('type', 'unknown')\n",
    }
    files.update({f"app/{name}.py": text for name, text in distractors.items()})
    return files


TASKS: tuple[CodingTask, ...] = (
    CodingTask(
        task_id="small-slug-bug",
        title="small localized bug",
        prompt=(
            "Fix the slugify bug in this small Python package. Punctuation should be "
            "removed, repeated separators collapsed, and the existing tests must pass. "
            "Keep the public function name and add a focused regression test if useful."
        ),
        files=_project(
            (
                "app/slug.py",
                """import re


def slugify(value: str) -> str:
    return re.sub(r"\\s+", "-", value.strip().lower())
""",
            ),
            (
                "tests/test_slug.py",
                """from app.slug import slugify


def test_slugify_removes_punctuation_and_collapses_separators() -> None:
    assert slugify("  Hello,  World! ") == "hello-world"
""",
            ),
        ),
        must_change=("app/slug.py",),
    ),
    CodingTask(
        task_id="medium-timezone-investigation",
        title="medium timezone investigation",
        prompt=(
            "Investigate and fix the event timestamp bug. ISO-8601 values ending in Z "
            "must become timezone-aware UTC datetimes, while explicit offsets must be "
            "normalized to UTC. Preserve the public API and make the tests pass."
        ),
        files=_project(
            (
                "app/events.py",
                """from datetime import datetime


def parse_event_time(value: str) -> datetime:
    # The original implementation discarded timezone information.
    return datetime.fromisoformat(value.replace("Z", ""))
""",
            ),
            (
                "app/ingest.py",
                """from datetime import datetime
from app.events import parse_event_time


def event_day(value: str) -> str:
    return parse_event_time(value).date().isoformat()
""",
            ),
            (
                "tests/test_events.py",
                """from datetime import timezone

from app.events import parse_event_time


def test_zulu_is_aware_utc() -> None:
    value = parse_event_time("2025-04-01T23:30:00Z")
    assert value.tzinfo == timezone.utc


def test_offset_is_normalized_to_utc() -> None:
    value = parse_event_time("2025-04-02T01:30:00+02:00")
    assert value.isoformat() == "2025-04-01T23:30:00+00:00"
""",
            ),
        ),
        must_change=("app/events.py",),
    ),
    CodingTask(
        task_id="large-candidate-triage",
        title="large candidate set",
        prompt=(
            "Fix the cart total defect and keep the package tests passing. There are many "
            "unrelated modules in this repository, so investigate efficiently and avoid "
            "unrelated edits. If you discover a large set of plausible candidates, you "
            "may use the available local laya-mcp tools for advisory triage, but decide "
            "yourself whether that is worthwhile."
        ),
        files=_large_candidate_project(),
        must_change=("app/cart.py",),
    ),
    CodingTask(
        task_id="test-selection",
        title="test selection and regression coverage",
        prompt=(
            "Fix the discount calculation so percentages are applied to cents without "
            "premature rounding. Add or update the smallest useful regression test, run "
            "the relevant tests, and avoid changing unrelated refund or inventory code. "
            "You may use laya-mcp for advisory triage only if it saves real work."
        ),
        files=_project(
            (
                "app/pricing.py",
                """def discounted_cents(price: int, percent: int) -> int:
    return price - (price // 100) * percent
""",
            ),
            ("app/refunds.py", "def refundable(cents: int) -> bool:\n    return cents > 0\n"),
            ("app/inventory.py", "def available(stock: int, reserved: int) -> int:\n    return max(stock - reserved, 0)\n"),
            (
                "tests/test_pricing.py",
                """from app.pricing import discounted_cents


def test_discount_keeps_fractional_cent_calculation_until_final_rounding() -> None:
    assert discounted_cents(999, 15) == 849
""",
            ),
            ("tests/test_refunds.py", "from app.refunds import refundable\n\ndef test_positive_is_refundable() -> None:\n    assert refundable(10)\n"),
            ("tests/test_inventory.py", "from app.inventory import available\n\ndef test_reserved_stock_is_unavailable() -> None:\n    assert available(5, 2) == 3\n"),
        ),
        must_change=("app/pricing.py",),
    ),
    CodingTask(
        task_id="error-investigation",
        title="error investigation",
        prompt=(
            "Investigate the failing request retry behavior. The client should retry a "
            "transient timeout twice with backoff, but should not retry a 4xx response. "
            "Use the supplied tests and error log as evidence, make the smallest correct "
            "fix, and run validation. Treat any local classifier suggestion as advisory."
        ),
        files=_project(
            (
                "app/client.py",
                """class HttpError(Exception):
    def __init__(self, status: int) -> None:
        self.status = status


def should_retry(error: Exception) -> bool:
    return isinstance(error, HttpError) and error.status >= 500


def request_with_retry(request: object, send: object) -> object:
    # Timeout errors are accidentally treated as permanent failures.
    for _attempt in range(2):
        try:
            return send(request)  # type: ignore[operator]
        except HttpError as error:
            if not should_retry(error):
                raise
    raise TimeoutError("request timed out")
""",
            ),
            (
                "logs/investigation.log",
                """2025-04-01 10:01 timeout while connecting to upstream
2025-04-01 10:02 404 missing optional avatar
2025-04-01 10:03 timeout while connecting to upstream
""",
            ),
            (
                "tests/test_client.py",
                """from app.client import HttpError, request_with_retry


def test_timeout_retries_then_succeeds() -> None:
    calls = 0

    def send(_request: object) -> object:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise TimeoutError("upstream timeout")
        return "ok"

    assert request_with_retry(object(), send) == "ok"
    assert calls == 3


def test_client_error_is_not_retried() -> None:
    calls = 0

    def send(_request: object) -> object:
        nonlocal calls
        calls += 1
        raise HttpError(404)

    try:
        request_with_retry(object(), send)
    except HttpError:
        pass
    assert calls == 1
""",
            ),
        ),
        must_change=("app/client.py",),
    ),
    CodingTask(
        task_id="refactor-settings",
        title="broader refactor",
        prompt=(
            "Refactor settings loading so callers receive a typed Settings dataclass "
            "instead of a loosely typed dictionary. Preserve environment-variable "
            "defaults and compatibility for the existing public load_settings function. "
            "Run all tests and avoid unrelated changes."
        ),
        files=_project(
            (
                "app/settings.py",
                """import os


def load_settings() -> dict[str, str]:
    return {
        "host": os.getenv("APP_HOST", "127.0.0.1"),
        "port": os.getenv("APP_PORT", "8080"),
    }
""",
            ),
            (
                "app/server.py",
                """from app.settings import load_settings


def address() -> str:
    settings = load_settings()
    return f"{settings['host']}:{settings['port']}"
""",
            ),
            (
                "tests/test_settings.py",
                """from app.server import address
from app.settings import load_settings


def test_defaults() -> None:
    assert address() == "127.0.0.1:8080"


def test_load_settings_returns_expected_values(monkeypatch) -> None:
    monkeypatch.setenv("APP_HOST", "0.0.0.0")
    monkeypatch.setenv("APP_PORT", "9000")
    assert load_settings() == {"host": "0.0.0.0", "port": "9000"}
""",
            ),
        ),
        must_change=("app/settings.py",),
    ),
    CodingTask(
        task_id="configuration-documentation",
        title="configuration and documentation",
        prompt=(
            "Add a documented APP_LOG_LEVEL configuration option with a safe INFO "
            "default. Update the example environment file and README, keep existing "
            "behavior, and add a focused test. This is a bounded documentation/config "
            "task; do not introduce a new dependency."
        ),
        files=_project(
            (
                "app/logging_config.py",
                """import os


def log_level() -> str:
    return os.getenv("APP_LOG_LEVEL", "INFO")
""",
            ),
            (
                ".env.example",
                """APP_HOST=127.0.0.1
""",
            ),
            (
                "README.md",
                """# Mini service

Run the tests with `python -m pytest -q`.
""",
            ),
            (
                "tests/test_logging_config.py",
                """from app.logging_config import log_level


def test_default_log_level() -> None:
    assert log_level() == "INFO"
""",
            ),
        ),
        validation=(
            "python",
            "-c",
            "from pathlib import Path; import subprocess; subprocess.run(['python','-m','pytest','-q'], check=True); assert 'APP_LOG_LEVEL' in Path('.env.example').read_text(); assert 'APP_LOG_LEVEL' in Path('README.md').read_text()",
        ),
        must_change=("app/logging_config.py", ".env.example", "README.md"),
    ),
    CodingTask(
        task_id="multi-step-order-validation",
        title="multi-step coding task",
        prompt=(
            "Implement the missing order validation workflow. Reject empty orders, reject "
            "negative quantities, calculate the total using the supplied price table, "
            "and preserve the existing OrderError API. Add focused tests and run the full "
            "suite. Use laya-mcp only for bounded advisory decisions if it clearly saves "
            "primary-model work."
        ),
        files=_project(
            (
                "app/orders.py",
                """class OrderError(ValueError):
    pass


def order_total(lines: list[tuple[str, int]], prices: dict[str, int]) -> int:
    # Validation and total calculation are intentionally incomplete.
    return sum(prices[name] for name, _quantity in lines)
""",
            ),
            (
                "app/checkout.py",
                """from app.orders import order_total


def checkout(lines: list[tuple[str, int]], prices: dict[str, int]) -> int:
    return order_total(lines, prices)
""",
            ),
            (
                "tests/test_orders.py",
                """import pytest

from app.orders import OrderError, order_total


def test_total_uses_quantity() -> None:
    assert order_total([("book", 2), ("pen", 3)], {"book": 1000, "pen": 200}) == 2600


def test_empty_order_is_rejected() -> None:
    with pytest.raises(OrderError):
        order_total([], {"book": 1000})


def test_negative_quantity_is_rejected() -> None:
    with pytest.raises(OrderError):
        order_total([("book", -1)], {"book": 1000})
""",
            ),
        ),
        must_change=("app/orders.py",),
    ),
)


if __name__ == "__main__":
    for task in TASKS:
        print(task.task_id)
