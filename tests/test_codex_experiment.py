from __future__ import annotations

from benchmarks.codex_task_suite import TASKS


def test_codex_task_suite_is_small_and_unique() -> None:
    assert 8 <= len(TASKS) <= 12
    ids = [task.task_id for task in TASKS]
    assert len(ids) == len(set(ids))
    assert {"large-candidate-triage", "error-investigation", "refactor-settings"} <= set(ids)


def test_codex_tasks_have_objective_validation_and_fixture_files() -> None:
    for task in TASKS:
        assert task.files["pyproject.toml"]
        assert task.validation
        assert task.must_change
