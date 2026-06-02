"""Unit tests for run.py orchestrator subcommands.

These tests use monkeypatching to avoid real subprocess calls or LLM traffic.
"""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run  # noqa: E402


def _write_json(p: Path, payload: dict) -> None:
    p.write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


def test_main_requires_subcommand():
    with pytest.raises(SystemExit):
        run.main()


def test_main_dispatches_validate(monkeypatch):
    called = {}
    monkeypatch.setattr(
        run, "cmd_validate", lambda args: (called.setdefault("validate", True), 0)[1]
    )
    monkeypatch.setattr("sys.argv", ["run.py", "validate"])
    rc = run.main()
    assert rc == 0
    assert called.get("validate") is True


def test_main_dispatches_retry_llm(monkeypatch):
    called = {}
    monkeypatch.setattr(
        run,
        "cmd_retry_llm",
        lambda args: (called.setdefault(args.dry_run, True), 0)[1],
    )
    monkeypatch.setattr("sys.argv", ["run.py", "retry-llm", "--dry-run"])
    rc = run.main()
    assert rc == 0
    assert called.get(True) is True


def test_main_dispatches_compare(monkeypatch):
    called = {}
    monkeypatch.setattr(
        run, "cmd_compare", lambda args: (called.setdefault("compare", True), 0)[1]
    )
    monkeypatch.setattr("sys.argv", ["run.py", "compare"])
    rc = run.main()
    assert rc == 0
    assert called.get("compare") is True


def test_main_dispatches_all(monkeypatch):
    called = {}
    monkeypatch.setattr(
        run,
        "cmd_all",
        lambda args: (called.setdefault("with_retry", args.with_retry), 0)[1],
    )
    monkeypatch.setattr("sys.argv", ["run.py", "all", "--with-retry"])
    rc = run.main()
    assert rc == 0
    assert called.get("with_retry") is True


# ---------------------------------------------------------------------------
# cmd_validate
# ---------------------------------------------------------------------------


def test_cmd_validate_reports_ok_and_errors(tmp_path, monkeypatch):
    # Build a fake data tree with 1 OK file and 1 error file.
    data_llm_dir = tmp_path / "data" / "llm_results" / "dictionary_llm_results"
    (data_llm_dir / "tab_a" / "json").mkdir(parents=True)
    (data_llm_dir / "tab_b" / "json").mkdir(parents=True)
    _write_json(
        data_llm_dir / "tab_a" / "json" / "openai_small_gpt-5.4-mini_parsed.json",
        {"table_name": "tab_a", "fields": []},
    )
    _write_json(
        data_llm_dir / "tab_b" / "json" / "openai_small_gpt-5.4-nano_parsed.json",
        {"error": "rate limit"},
    )

    # Use absolute path so the function's `PROJECT_ROOT / path` is a no-op
    # (Path drops the left side when right side is absolute).
    cfg = {
        "data_llm_results_dictionary_generation": {
            "path": str(data_llm_dir),
        }
    }
    monkeypatch.setattr(run, "load_config", lambda: cfg)

    rc = run.cmd_validate(argparse.Namespace())
    assert rc == 0


def test_cmd_validate_handles_malformed_json(tmp_path, monkeypatch):
    data_llm_dir = tmp_path / "data" / "llm_results" / "dictionary_llm_results"
    (data_llm_dir / "tab" / "json").mkdir(parents=True)
    (data_llm_dir / "tab" / "json" / "bad_parsed.json").write_text("{not valid")

    cfg = {
        "data_llm_results_dictionary_generation": {
            "path": str(data_llm_dir),
        }
    }
    monkeypatch.setattr(run, "load_config", lambda: cfg)

    rc = run.cmd_validate(argparse.Namespace())
    assert rc == 0


# ---------------------------------------------------------------------------
# cmd_compare (summary printing only)
# ---------------------------------------------------------------------------


def test_cmd_compare_prints_summary_on_success(tmp_path, monkeypatch):
    summary = {
        "results": {"tab1": {}, "tab2": {}},
        "average_by_model": {
            "deepseek-v4-flash": 0.61,
            "gpt-5.4-mini": 0.65,
        },
    }
    out_dir = tmp_path / "data" / "distance_calculation"
    out_dir.mkdir(parents=True)
    (out_dir / "all_similarities_results.json").write_text(json.dumps(summary))

    fake_proc = MagicMock(returncode=0)
    monkeypatch.setattr(run.subprocess, "call", lambda *a, **kw: fake_proc.returncode)

    rc = run.cmd_compare(argparse.Namespace())
    assert rc == 0


def test_cmd_compare_skips_summary_when_subprocess_fails(tmp_path, monkeypatch):
    fake_proc = MagicMock(returncode=1)
    monkeypatch.setattr(run.subprocess, "call", lambda *a, **kw: fake_proc.returncode)

    rc = run.cmd_compare(argparse.Namespace())
    assert rc == 1
    # And no crash if the summary file is missing.


# ---------------------------------------------------------------------------
# cmd_retry_llm
# ---------------------------------------------------------------------------


def test_cmd_retry_llm_dry_run_uses_list_errors(monkeypatch):
    captured = {}
    fake_proc = MagicMock(returncode=0)

    def fake_call(cmd, **kw):
        captured["cmd"] = cmd
        return fake_proc.returncode

    monkeypatch.setattr(run.subprocess, "call", fake_call)
    monkeypatch.setattr(run, "find_python", lambda: "/fake/python")

    rc = run.cmd_retry_llm(argparse.Namespace(dry_run=True))
    assert rc == 0
    assert "--list-errors" in captured["cmd"]
    assert "--retry-errors" not in captured["cmd"]


def test_cmd_retry_llm_real_uses_retry_errors(monkeypatch):
    captured = {}
    fake_proc = MagicMock(returncode=0)

    def fake_call(cmd, **kw):
        captured["cmd"] = cmd
        return fake_proc.returncode

    monkeypatch.setattr(run.subprocess, "call", fake_call)
    monkeypatch.setattr(run, "find_python", lambda: "/fake/python")

    rc = run.cmd_retry_llm(argparse.Namespace(dry_run=False))
    assert rc == 0
    assert "--retry-errors" in captured["cmd"]
    assert "--list-errors" not in captured["cmd"]


# ---------------------------------------------------------------------------
# cmd_all (orchestration)
# ---------------------------------------------------------------------------


def test_cmd_all_runs_validate_then_compare(monkeypatch):
    calls = []
    monkeypatch.setattr(
        run, "cmd_validate", lambda a: (calls.append("validate") or 0)
    )
    monkeypatch.setattr(
        run, "cmd_compare", lambda a: (calls.append("compare") or 0)
    )
    monkeypatch.setattr(
        run, "cmd_retry_llm", lambda a: (calls.append("retry") or 0)
    )
    rc = run.cmd_all(argparse.Namespace(with_retry=False))
    assert rc == 0
    assert calls == ["validate", "compare"]


def test_cmd_all_with_retry_includes_retry_step(monkeypatch):
    calls = []
    monkeypatch.setattr(
        run, "cmd_validate", lambda a: (calls.append("validate") or 0)
    )
    monkeypatch.setattr(
        run, "cmd_compare", lambda a: (calls.append("compare") or 0)
    )
    monkeypatch.setattr(
        run, "cmd_retry_llm", lambda a: (calls.append("retry") or 0)
    )
    rc = run.cmd_all(argparse.Namespace(with_retry=True))
    assert rc == 0
    assert calls == ["validate", "retry", "compare"]


def test_cmd_all_short_circuits_on_validate_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(run, "cmd_validate", lambda a: (calls.append("v") or 1))
    monkeypatch.setattr(run, "cmd_compare", lambda a: (calls.append("c") or 0))
    monkeypatch.setattr(run, "cmd_retry_llm", lambda a: (calls.append("r") or 0))
    rc = run.cmd_all(argparse.Namespace(with_retry=True))
    assert rc == 1
    assert "c" not in calls
    assert "r" not in calls
