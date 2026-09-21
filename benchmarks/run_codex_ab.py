#!/usr/bin/env python3
"""Run reproducible vanilla-Codex versus Codex+laya-mcp task comparisons.

The runner creates a fresh temporary Git repository for every condition, invokes
the installed Codex CLI, runs objective validation outside the agent, and writes
compact JSON results plus raw Codex JSONL transcripts.  It deliberately does not
tell Codex to call a particular laya-mcp tool.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any

if __package__ in {None, ""}:  # support direct execution from the repository root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.codex_task_suite import CodingTask, TASKS


REPO_ROOT = Path(__file__).resolve().parents[1]
LAYA_EXECUTABLE = REPO_ROOT / ".venv" / "bin" / "laya-mcp"
W8_MODEL = "aac6fef/laya-multilingual-coreml-ane-w8"
BEHAVIOR_GUIDANCE = """You have access to laya-mcp, a local advisory MCP server.
Use it only when a cheap bounded decision may avoid unnecessary primary-model
reasoning or context. It is appropriate for simple binary/choice decisions and
advisory triage of already-discovered candidates. Do not use it for architecture,
complex debugging, security-sensitive or destructive authorization, code generation,
nuanced semantic reasoning, or final correctness judgments. Its confidence is not a
calibrated probability. `filter` is advisory: never treat excluded candidates as
unavailable, and inspect any candidate when evidence or task importance warrants it.
Do not call a tool merely to demonstrate the integration; decide whether the call is
actually worth its overhead, and continue reasoning yourself when uncertain.
"""


def _run(command: list[str], *, cwd: Path, timeout: float = 600.0) -> tuple[int, str, str, float]:
    started = perf_counter()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        stdout = stdout or str(exc.stdout or "")
        stderr = stderr or str(exc.stderr or "")
        return 124, stdout, stderr, (perf_counter() - started) * 1000
    return process.returncode, stdout, stderr, (perf_counter() - started) * 1000


def _write_workspace(task: CodingTask, root: Path) -> None:
    for relative, content in task.files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _run(["git", "init", "-q"], cwd=root)
    _run(["git", "config", "user.email", "codex-eval@example.invalid"], cwd=root)
    _run(["git", "config", "user.name", "Codex evaluation"], cwd=root)
    _run(["git", "add", "."], cwd=root)
    _run(["git", "commit", "-qm", "fixture"], cwd=root)


def _jsonl(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def _tool_payload(item: dict[str, Any]) -> dict[str, Any] | None:
    result = item.get("result")
    if not isinstance(result, dict):
        return None
    structured = result.get("structured_content")
    if isinstance(structured, dict):
        return structured
    for content in result.get("content", []):
        if not isinstance(content, dict) or content.get("type") != "text":
            continue
        try:
            parsed = json.loads(str(content.get("text", "")))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _codex_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    tools: list[dict[str, Any]] = []
    commands: list[str] = []
    for event in events:
        if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = event["usage"]
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "mcp_tool_call":
            payload = _tool_payload(item)
            tool = {
                "server": item.get("server"),
                "tool": item.get("tool"),
                "status": item.get("status"),
                "error": item.get("error"),
                "arguments": item.get("arguments"),
                "payload": payload,
            }
            tools.append(tool)
        if item_type == "command_execution":
            command = item.get("command")
            if isinstance(command, str):
                commands.append(command)
    tool_counts = Counter(str(tool["tool"]) for tool in tools)
    laya_input = 0
    laya_output = 0
    laya_latency = 0.0
    laya_evaluations = 0
    for tool in tools:
        payload = tool.get("payload")
        if not isinstance(payload, dict):
            continue
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            laya_input += int(metrics.get("total_input_tokens", metrics.get("input_tokens", 0)) or 0)
            laya_output += int(metrics.get("total_output_tokens", metrics.get("output_tokens", 0)) or 0)
            laya_latency += float(
                metrics.get("total_inference_latency_ms", metrics.get("inference_latency_ms", 0.0)) or 0.0
            )
            laya_evaluations += int(metrics.get("model_evaluations", 1) or 0)
        elif "input_tokens" in payload:
            laya_input += int(payload.get("input_tokens", 0) or 0)
            laya_output += int(payload.get("output_tokens", 0) or 0)
            laya_latency += float(payload.get("inference_latency_ms", 0.0) or 0.0)
            laya_evaluations += 1
    search_commands = [
        command for command in commands if any(word in command.split()[:1] for word in ("rg", "grep", "find"))
    ]
    read_commands = [
        command
        for command in commands
        if any(command.lstrip().startswith(word) for word in ("cat", "sed", "head", "tail", "awk"))
    ]
    return {
        "usage": usage,
        "mcp_call_count": len(tools),
        "mcp_tool_counts": dict(tool_counts),
        "mcp_tools": tools,
        "commands": commands,
        "search_command_count": len(search_commands),
        "read_command_count": len(read_commands),
        "laya_input_tokens": laya_input,
        "laya_output_tokens": laya_output,
        "laya_inference_latency_ms": round(laya_latency, 3),
        "laya_model_evaluations": laya_evaluations,
    }


def _codex_command(condition: str, workspace: Path, model: str | None) -> list[str]:
    command = [
        "codex",
        "exec",
        "--ephemeral",
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "-C",
        str(workspace),
    ]
    if model:
        command.extend(["-m", model])
    if condition == "laya":
        command.extend(
            [
                "-c",
                f'mcp_servers.laya.command="{LAYA_EXECUTABLE}"',
                "-c",
                "mcp_servers.laya.startup_timeout_sec=90",
                "-c",
                "mcp_servers.laya.tool_timeout_sec=120",
                "-c",
                "mcp_servers.laya.required=true",
                "-c",
                f'mcp_servers.laya.env={{LAYA_MCP_MODEL="{W8_MODEL}",LAYA_MCP_LOCAL_FILES_ONLY="true"}}',
            ]
        )
    return command


def _prompt(task: CodingTask, condition: str) -> str:
    if condition == "laya":
        return f"{BEHAVIOR_GUIDANCE}\n\nTask:\n{task.prompt}"
    return task.prompt


def _validate(task: CodingTask, workspace: Path) -> dict[str, Any]:
    code, stdout, stderr, elapsed_ms = _run(list(task.validation), cwd=workspace, timeout=120)
    _, changed_stdout, _, _ = _run(["git", "diff", "--name-only"], cwd=workspace)
    changed = [line for line in changed_stdout.splitlines() if line]
    changed_required = [path for path in task.must_change if path in changed]
    return {
        "success": code == 0 and all(path in changed for path in task.must_change),
        "validation_exit_code": code,
        "validation_stdout": stdout[-12000:],
        "validation_stderr": stderr[-12000:],
        "validation_latency_ms": round(elapsed_ms, 3),
        "changed_files": changed,
        "required_files_changed": changed_required,
        "missing_required_files": [path for path in task.must_change if path not in changed],
    }


def run_one(task: CodingTask, condition: str, *, output_dir: Path, model: str | None) -> dict[str, Any]:
    workspace = Path(tempfile.mkdtemp(prefix=f"laya-codex-{task.task_id}-{condition}-"))
    raw_path = output_dir / f"{task.task_id}-{condition}.jsonl"
    started = perf_counter()
    try:
        _write_workspace(task, workspace)
        command = _codex_command(condition, workspace, model)
        command.append(_prompt(task, condition))
        try:
            code, stdout, stderr, _ = _run(command, cwd=workspace, timeout=600)
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            code = 124
            stdout = str(exc.stdout or "")
            stderr = str(exc.stderr or "")
            timed_out = True
        raw_path.write_text(stdout, encoding="utf-8")
        events = _jsonl(stdout)
        metrics = _codex_metrics(events)
        validation = _validate(task, workspace) if not timed_out else {"success": False}
        return {
            "task_id": task.task_id,
            "title": task.title,
            "condition": condition,
            "model": model,
            "laya_model": W8_MODEL if condition == "laya" else None,
            "codex_exit_code": code,
            "timed_out": timed_out,
            "elapsed_ms": round((perf_counter() - started) * 1000, 3),
            "stderr_tail": stderr[-12000:],
            "raw_event_file": str(raw_path),
            **metrics,
            **validation,
        }
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("vanilla", "laya", "both"), default="both")
    parser.add_argument("--task", action="append", dest="tasks", help="task ID; repeatable")
    parser.add_argument("--model", help="explicit Codex model; omit to use Codex default")
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/codex-ab/results.json"))
    args = parser.parse_args()
    selected = [task for task in TASKS if not args.tasks or task.task_id in args.tasks]
    if not selected:
        raise SystemExit("no matching tasks")
    conditions = ("vanilla", "laya") if args.condition == "both" else (args.condition,)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = [
        run_one(task, condition, output_dir=args.output.parent, model=args.model)
        for task in selected
        for condition in conditions
    ]
    args.output.write_text(
        json.dumps(
            {
                "codex_version": subprocess.run(["codex", "--version"], capture_output=True, text=True, check=False).stdout.strip(),
                "laya_executable": str(LAYA_EXECUTABLE),
                "laya_model": W8_MODEL,
                "tasks": [task.task_id for task in selected],
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "result_count": len(results)}, indent=2))


if __name__ == "__main__":
    main()
