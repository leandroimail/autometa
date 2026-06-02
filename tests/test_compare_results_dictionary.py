"""Unit tests for src/compare_results_dictionary.py.

Focuses on the fixed behaviors:
  * model_name extraction via known client prefixes (Fix A)
  * table_name extraction with the new data_llm_dir layout (Fix B + C)
  * filtering of LLM error results in the load loop (Fix D)
  * JSON / CSV loaders
  * similarity / aggregation math (small inputs, no model download)
"""

import json
import math
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"
FIXTURES_DIR = TESTS_DIR / "fixtures"

sys.path.insert(0, str(SRC))

from compare_results_dictionary import (  # noqa: E402
    extract_table_and_model_names,
    find_all_files_in_directory,
    generate_output_json,
    load_csv_data,
    load_data_dictionary,
    load_json_data,
    calculate_similarities,
)


# ---------------------------------------------------------------------------
# extract_table_and_model_names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data_llm_dir,rel_path,expected_table,expected_model",
    [
        (
            "data/llm_results/dictionary_llm_results",
            "data/llm_results/dictionary_llm_results/bird__california_schools__frpm/json/openai_small_gpt-5.4-mini_parsed.json",
            "bird__california_schools__frpm",
            "gpt-5.4-mini",
        ),
        (
            "data/llm_results/dictionary_llm_results",
            "data/llm_results/dictionary_llm_results/bird__toxicology__bond/json/deepseek_small_deepseek-v4-flash_parsed.json",
            "bird__toxicology__bond",
            "deepseek-v4-flash",
        ),
        (
            "data/llm_results/dictionary_llm_results",
            "data/llm_results/dictionary_llm_results/bird__codebase_community__users/json/google_small_gemini-3.5-flash_parsed.json",
            "bird__codebase_community__users",
            "gemini-3.5-flash",
        ),
        (
            "data/llm_results/dictionary_llm_results",
            "data/llm_results/dictionary_llm_results/bird__codebase_community__users/json/google_small_gemini-3.1-flash-lite_parsed.json",
            "bird__codebase_community__users",
            "gemini-3.1-flash-lite",
        ),
        (
            "data/llm_results/dictionary_llm_results",
            "data/llm_results/dictionary_llm_results/bird__european_football_2__team/json/deepseek_small_deepseek-v4-pro_parsed.json",
            "bird__european_football_2__team",
            "deepseek-v4-pro",
        ),
        (
            "data/llm_results/dictionary_llm_results",
            "data/llm_results/dictionary_llm_results/bird__superhero__hero/json/openai_small_gpt-5.4-nano_parsed.json",
            "bird__superhero__hero",
            "gpt-5.4-nano",
        ),
    ],
)
def test_extract_table_and_model_names_six_models(
    data_llm_dir, rel_path, expected_table, expected_model
):
    table, model = extract_table_and_model_names(data_llm_dir, rel_path)
    assert table == expected_table
    assert model == expected_model


def test_extract_table_and_model_names_unrecognised_prefix_falls_back_to_stem():
    """Unknown client prefix should fall back to the file stem (post _parsed)."""
    table, model = extract_table_and_model_names(
        "data/llm_results/dictionary_llm_results",
        "data/llm_results/dictionary_llm_results/some__table/json/mystery_unknown-model_parsed.json",
    )
    assert table == "some__table"
    # Unknown prefix: the whole stem (minus _parsed) is returned.
    assert model == "mystery_unknown-model"


def test_extract_table_and_model_names_uses_index_one_with_new_layout():
    """Regression: the old code used split[2] which became 'json' under the new path."""
    table, _ = extract_table_and_model_names(
        "data/llm_results/dictionary_llm_results",
        "data/llm_results/dictionary_llm_results/bird__formula_1__races/json/openai_small_gpt-5.4-mini_parsed.json",
    )
    assert table != "json"
    assert table == "bird__formula_1__races"


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def test_load_json_data_returns_dict(tmp_path):
    p = tmp_path / "ok.json"
    p.write_text(json.dumps({"a": 1, "b": [1, 2]}))
    assert load_json_data(str(p)) == {"a": 1, "b": [1, 2]}


def test_load_json_data_returns_empty_on_error(tmp_path, caplog):
    p = tmp_path / "bad.json"
    p.write_text("{not valid")
    with caplog.at_level("ERROR", logger="__main__"):
        result = load_json_data(str(p))
    assert result == {}


def test_load_csv_data_reads_field_name_and_description(tmp_path):
    p = tmp_path / "dict.csv"
    p.write_text(
        "field_name,description\n"
        "id,Primary key\n"
        "name,Display name\n"
    )
    assert load_csv_data(str(p)) == {"id": "Primary key", "name": "Display name"}


def test_load_data_dictionary_dispatches_by_extension(tmp_path):
    json_p = tmp_path / "d.json"
    json_p.write_text(json.dumps({"id": "pk"}))
    csv_p = tmp_path / "d.csv"
    csv_p.write_text("field_name,description\nid,Primary key\n")

    assert load_data_dictionary(str(json_p)) == {"id": "pk"}
    assert load_data_dictionary(str(csv_p)) == {"id": "Primary key"}


def test_load_data_dictionary_unsupported_extension_returns_empty(tmp_path, caplog):
    p = tmp_path / "d.txt"
    p.write_text("anything")
    with caplog.at_level("WARNING", logger="__main__"):
        assert load_data_dictionary(str(p)) == {}


def test_find_all_files_in_directory_recursive(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.json").write_text("{}")
    (tmp_path / "b.json").write_text("{}")
    (tmp_path / "ignore.xml").write_text("<x/>")

    files = find_all_files_in_directory(str(tmp_path), [".json", ".csv"])
    assert len(files) == 2
    assert any(f.endswith("x.json") for f in files)
    assert any(f.endswith("b.json") for f in files)


# ---------------------------------------------------------------------------
# calculate_similarities (uses real numpy + sklearn; no model download)
# ---------------------------------------------------------------------------


def _vec(values):
    return np.array(values, dtype=float)


def test_calculate_similarities_identical_vectors_score_one():
    a = _vec([1, 0, 0])
    b = _vec([1, 0, 0])
    sims = calculate_similarities({"x": a}, {"x": b})
    assert sims == [{"field": "x", "score": pytest.approx(1.0)}]


def test_calculate_similarities_orthogonal_vectors_score_zero():
    a = _vec([1, 0])
    b = _vec([0, 1])
    sims = calculate_similarities({"x": a}, {"x": b})
    assert sims[0]["score"] == pytest.approx(0.0)


def test_calculate_similarities_ignores_missing_keys():
    a = _vec([1, 0])
    sims = calculate_similarities({"x": a}, {"y": _vec([1, 0])})
    assert sims == []


def test_calculate_similarities_scores_in_minus_one_one_range():
    a = _vec([1, 2, 3, 4])
    b = _vec([-1, -2, -3, -4])
    sims = calculate_similarities({"f": a}, {"f": b})
    assert sims[0]["score"] == pytest.approx(-1.0)
    assert -1.0 <= sims[0]["score"] <= 1.0


# ---------------------------------------------------------------------------
# generate_output_json
# ---------------------------------------------------------------------------


def test_generate_output_json_shape():
    out = generate_output_json(
        table_name="t",
        model_name="m",
        similarities=[{"field": "f", "score": 0.9}],
    )
    assert out == {
        "table_name": "t",
        "par-compare-models": "m",
        "similarities": [{"field": "f", "score": 0.9}],
    }


# ---------------------------------------------------------------------------
# End-to-end with a fake SentenceTransformer (no model download)
# ---------------------------------------------------------------------------


class _FakeModel:
    """Returns a deterministic embedding based on the input string length."""

    def encode(self, text):
        if not isinstance(text, str) or not text.strip():
            return np.zeros(4, dtype=float)
        # Two identical texts -> same vector; different texts -> orthogonal vectors.
        h = abs(hash(text)) % 97
        v = np.zeros(4, dtype=float)
        v[h % 4] = 1.0
        if (h // 4) % 4 != h % 4:
            v[(h // 4) % 4] = 1.0
        return v


def test_main_end_to_end_with_fake_model(tmp_path, monkeypatch):
    """Drive main() with a fake embedding model and a tiny filesystem."""
    import re
    cfg_src = PROJECT_ROOT / "config.yaml"
    cfg = cfg_src.read_text(encoding="utf-8")

    table = "bird__unit__test"
    # Build a fake data tree inside tmp_path.
    dict_dir = tmp_path / "data" / "dictionaries"
    llm_root = tmp_path / "data" / "llm_results" / "dictionary_llm_results" / table / "json"
    out_dir = tmp_path / "data" / "distance_calculation"
    dict_dir.mkdir(parents=True)
    llm_root.mkdir(parents=True)
    out_dir.mkdir(parents=True)

    # Baseline dictionary.
    (dict_dir / f"{table}.json").write_text(
        json.dumps(
            {
                "table_name": table,
                "table_description": "Baseline desc",
                "fields": [
                    {"field_name": "id", "field_description": "Primary key"},
                    {"field_name": "name", "field_description": "Display name"},
                ],
            }
        )
    )

    # Two LLM dictionaries, plus one with an error key (must be skipped).
    (llm_root / "openai_small_gpt-5.4-mini_parsed.json").write_text(
        json.dumps(
            {
                "table_name": table,
                "table_description": "LLM desc",
                "fields": [
                    {"field_name": "id", "full_description": "Primary key"},
                    {"field_name": "name", "full_description": "Display name"},
                ],
            }
        )
    )
    (llm_root / "deepseek_small_deepseek-v4-flash_parsed.json").write_text(
        json.dumps(
            {
                "table_name": table,
                "table_description": "LLM desc alt",
                "fields": [
                    {"field_name": "id", "full_description": "Primary key"},
                    {"field_name": "name", "full_description": "Display name"},
                ],
            }
        )
    )
    (llm_root / "google_small_gemini-3.5-flash_parsed.json").write_text(
        json.dumps({"error": "rate limit"})
    )

    # Build a minimal config that points to our temp tree.
    # data_llm_dir must point to the *parent* of the per-table directories,
    # matching the production layout (data/llm_results/dictionary_llm_results).
    new_cfg = (
        "list_of_data_dictionaries:\n"
        f"- name: {table}\n"
        f"  path: {dict_dir / (table + '.json')}\n"
        "data_llm_results_dictionary_generation:\n"
        f"  path: {llm_root.parent.parent}\n"
        "data_llm_results_distance_calculation:\n"
        f"  path: {out_dir}\n"
    )

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(new_cfg)

    from compare_results_dictionary import main

    # Patch the embedding model.
    monkeypatch.setattr(
        "compare_results_dictionary.SentenceTransformer", lambda *a, **kw: _FakeModel()
    )

    main(str(cfg_path))

    # Verify outputs: 2 LLM files survived (error one skipped), 1 baseline.
    out_files = list(out_dir.glob("output_*.json"))
    assert len(out_files) == 2
    summary = json.loads((out_dir / "all_similarities_results.json").read_text())
    assert table in summary["results"]
    # Exactly 2 models should be present (the error one must be filtered).
    models = set(summary["average_by_model"].keys())
    assert models == {"gpt-5.4-mini", "deepseek-v4-flash"}
    # No scores should be NaN — fake embeddings give valid cosine similarities.
    for field_scores in summary["results"][table].values():
        for score in field_scores.values():
            assert not math.isnan(score)
            assert -1.0 <= score <= 1.0
