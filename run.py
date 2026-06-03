#!/usr/bin/env python3
"""Orchestrator for the schema_generator pipeline.

This is the single entry point that wires every step of the pipeline together:

  * ``bootstrap``   - ``src/bootstrap_bird_mini_dev.py``
  * ``generate``    - ``src/dictionary_generation.py``
  * ``validate``    - audit parsed LLM JSONs without calling any LLM
  * ``retry-llm``   - rerun only failed LLM dictionary generations
  * ``compare``     - ``src/compare_results_dictionary.py``
  * ``metrics``     - print the consolidated metrics summary
  * ``all``         - validate [+retry-llm] + compare
  * ``pipeline``    - bootstrap -> generate -> all

Every external script is invoked as a subprocess so failures propagate as exit
codes and there is no need to reimplement its CLI in this file.
"""

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    import yaml
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def find_python() -> str:
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    return str(venv_python) if venv_python.exists() else sys.executable


def _run_subprocess(cmd, **kw) -> int:
    print(">>>", " ".join(cmd))
    return subprocess.call(cmd, cwd=PROJECT_ROOT, **kw)


def cmd_bootstrap(args) -> int:
    """Invoke ``src/bootstrap_bird_mini_dev.run`` to materialise BIRD Mini-Dev."""
    py = find_python()
    cmd = [py, "src/bootstrap_bird_mini_dev.py"]
    if args.update_config:
        cmd.append("--update-config")
    if args.dataset_url:
        cmd.extend(["--dataset-url", args.dataset_url])
    if args.sample_size is not None:
        cmd.extend(["--sample-size", str(args.sample_size)])
    if args.profile_sample_size is not None:
        cmd.extend(["--profile-sample-size", str(args.profile_sample_size)])
    return _run_subprocess(cmd)


def cmd_generate(args) -> int:
    """Invoke ``src/dictionary_generation.generate_dictionaries``."""
    py = find_python()
    cmd = [py, "src/dictionary_generation.py"]
    if args.retry_errors:
        cmd.append("--retry-errors")
    elif args.list_errors:
        cmd.append("--list-errors")
    return _run_subprocess(cmd)


def cmd_validate(args) -> int:
    cfg = load_config()
    data_llm_dir = PROJECT_ROOT / cfg.get("data_llm_results_dictionary_generation", {}).get("path", "")
    files = sorted(glob.glob(str(data_llm_dir / "**" / "*_parsed.json"), recursive=True))
    print(f"Found {len(files)} parsed JSON files under {data_llm_dir}")

    ok = err = 0
    err_samples = []
    for p in files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict) and d.get("error"):
                err += 1
                if len(err_samples) < 10:
                    err_samples.append((p, str(d.get("error"))[:120]))
            else:
                ok += 1
        except Exception as e:
            err += 1
            if len(err_samples) < 10:
                err_samples.append((p, f"parse error: {e}"))

    print(f"OK:  {ok}")
    print(f"ERR: {err}")
    if err_samples:
        print("\nFirst errors:")
        for path, msg in err_samples:
            p = Path(path)
            try:
                display = p.relative_to(PROJECT_ROOT)
            except ValueError:
                display = p
            print(f"  {display}: {msg}")
    return 0


def cmd_retry_llm(args) -> int:
    py = find_python()
    cmd = [py, "src/dictionary_generation.py"]
    if args.dry_run:
        cmd.append("--list-errors")
    else:
        cmd.append("--retry-errors")
    return _run_subprocess(cmd)


def cmd_compare(args) -> int:
    py = find_python()
    cmd = [py, "src/compare_results_dictionary.py"]
    rc = _run_subprocess(cmd)
    if rc == 0:
        summary_path = PROJECT_ROOT / "data" / "distance_calculation" / "all_similarities_results.json"
        if summary_path.exists():
            with open(summary_path, encoding="utf-8") as f:
                d = json.load(f)
            print("\n=== Summary ===")
            print(f"Tables: {len(d.get('results', {}))}")
            metrics = d.get("metrics_by_model", {})
            print(f"Models ({len(metrics)}): {sorted(metrics.keys())}")
            for m, s in sorted(metrics.items()):
                if isinstance(s, dict) and "mean" in s:
                    print(
                        f"  {m}: mean={s['mean']:.4f} "
                        f"std={s.get('std', 0):.4f} "
                        f"median={s.get('median', 0):.4f} "
                        f"min={s.get('min', 0):.4f} "
                        f"max={s.get('max', 0):.4f}"
                    )
                else:
                    print(f"  {m}: {s}")
    return rc


def cmd_metrics(args) -> int:
    """Print the consolidated metrics summary without recomputing distances."""
    summary_path = PROJECT_ROOT / "data" / "distance_calculation" / "all_similarities_results.json"
    if not summary_path.exists():
        print(f"Summary not found at {summary_path}. Run 'compare' first.")
        return 1

    with open(summary_path, encoding="utf-8") as f:
        d = json.load(f)

    print("=== Metrics by model ===")
    metrics_by_model = d.get("metrics_by_model", {})
    if not metrics_by_model:
        print("(no metrics_by_model entry in summary)")
    for model_name, metrics in sorted(metrics_by_model.items()):
        if not isinstance(metrics, dict):
            print(f"  {model_name}: {metrics}")
            continue
        ordered_keys = ["mean", "std", "q25", "median", "q75", "d90", "d99", "min", "max", "count"]
        parts = [f"{k}={metrics[k]:.4f}" for k in ordered_keys if k in metrics and k != "count"]
        if "count" in metrics:
            parts.append(f"n={metrics['count']}")
        print(f"  {model_name}: " + " ".join(parts))

    return 0


def cmd_all(args) -> int:
    rc = cmd_validate(args)
    if rc != 0:
        return rc
    if args.with_retry:
        rc = cmd_retry_llm(argparse.Namespace(dry_run=False))
        if rc != 0:
            return rc
    return cmd_compare(args)


def cmd_pipeline(args) -> int:
    """Run the full pipeline: bootstrap -> generate -> validate -> [retry-llm] -> compare."""
    if args.with_bootstrap:
        print("=== Step 1/3: bootstrap BIRD Mini-Dev ===")
        rc = cmd_bootstrap(
            argparse.Namespace(
                update_config=True,
                dataset_url=None,
                sample_size=None,
                profile_sample_size=None,
            )
        )
        if rc != 0:
            return rc

    if args.with_generate:
        print("=== Step 2/3: generate LLM dictionaries ===")
        rc = cmd_generate(argparse.Namespace(retry_errors=False, list_errors=False))
        if rc != 0:
            return rc

    print("=== Step 3/3: validate, optional retry, compare ===")
    return cmd_all(argparse.Namespace(with_retry=args.with_retry))


def main() -> int:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Schema generator orchestrator. Wraps every step of the pipeline.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    bootstrap_p = sub.add_parser(
        "bootstrap",
        help="Bootstrap BIRD Mini-Dev dictionaries, samples, and profiles",
    )
    bootstrap_p.add_argument(
        "--update-config",
        action="store_true",
        help="Refresh config.yaml with the new BIRD artifacts",
    )
    bootstrap_p.add_argument(
        "--dataset-url",
        default=None,
        help="Override the BIRD Mini-Dev download URL",
    )
    bootstrap_p.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Rows to include in the prompt sample (default: 100)",
    )
    bootstrap_p.add_argument(
        "--profile-sample-size",
        type=int,
        default=None,
        help="Rows to include in the profiling sample (default: 10000)",
    )
    bootstrap_p.set_defaults(func=cmd_bootstrap)

    generate_p = sub.add_parser(
        "generate",
        help="Generate LLM dictionaries for every configured profile",
    )
    generate_p.add_argument(
        "--retry-errors",
        action="store_true",
        help="Rerun only the provider/profile combinations that failed validation",
    )
    generate_p.add_argument(
        "--list-errors",
        action="store_true",
        help="List the provider/profile combinations that would be rerun",
    )
    generate_p.set_defaults(func=cmd_generate)

    sub.add_parser(
        "validate",
        help="Audit LLM result JSONs for errors (no API calls)",
    ).set_defaults(func=cmd_validate)

    retry = sub.add_parser(
        "retry-llm",
        help="Rerun failed LLM dictionary generations",
    )
    retry.add_argument(
        "--dry-run",
        action="store_true",
        help="List retry targets without calling LLM APIs",
    )
    retry.set_defaults(func=cmd_retry_llm)

    sub.add_parser(
        "compare",
        help="Run distance comparison against LLM dictionaries",
    ).set_defaults(func=cmd_compare)

    sub.add_parser(
        "metrics",
        help="Print the consolidated metrics summary from all_similarities_results.json",
    ).set_defaults(func=cmd_metrics)

    all_p = sub.add_parser(
        "all",
        help="validate [+retry-llm] + compare",
    )
    all_p.add_argument(
        "--with-retry",
        action="store_true",
        help="Include the retry-llm step before compare",
    )
    all_p.set_defaults(func=cmd_all)

    pipeline_p = sub.add_parser(
        "pipeline",
        help="Full pipeline: bootstrap -> generate -> validate -> [retry-llm] -> compare",
    )
    pipeline_p.add_argument(
        "--with-bootstrap",
        action="store_true",
        help="Run the bootstrap step first (downloads BIRD Mini-Dev if needed)",
    )
    pipeline_p.add_argument(
        "--with-generate",
        action="store_true",
        help="Run the LLM dictionary generation step",
    )
    pipeline_p.add_argument(
        "--with-retry",
        action="store_true",
        help="Include retry-llm before compare",
    )
    pipeline_p.set_defaults(func=cmd_pipeline)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
