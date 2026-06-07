# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Project rebrand to AutoMeta.** The project is now named `AutoMeta` across
  the codebase: `pyproject.toml` `name` is now `autometa` (with a non-empty
  `description`), the README title is `# AutoMeta`, and the technical articles
  have been renamed to `reports/AUTOMETA_ARTICLE.md` and
  `reports/AUTOMETA_ARTICLE.en.md` (with cross-references updated). The
  underlying capabilities (data dictionary generation, schema matching,
  similarity evaluation) are unchanged.

### Fixed
- **Typo `schema_mathcing` in `src/load_data.py`.** Renamed the local DuckDB
  database from `schema_mathcing` to `schema_matching` (file path, `db_name`,
  shell snippets, and the warning comment in the generated DDL doc).

### Added
- **Configurable embedding model.** The `sentence-transformers` model used by
  `src/compare_results_dictionary.py` is now driven by a new `embedding:` block
  in `config.yaml` (line ~838), read by the helper
  `_load_embedding_model(config)`. Defaults preserve the previous
  `all-MiniLM-L6-v2` behavior; the recommended override is
  `BAAI/bge-small-en-v1.5` (justified in `reports/EMBEDDING_MODEL.md`).
  New fields: `model_name`, `device`, `cache_dir`, `normalize_embeddings`,
  `batch_size`. `normalize_embeddings` and `batch_size` are now actually
  forwarded to every `model.encode(...)` call.

### Changed
- **Breaking — `all_similarities_results.json` schema.** The keys
  `average_by_model` and `average_by_table_and_model` (each
  `Dict[str, float]`) were replaced by `metrics_by_model` and
  `metrics_by_table_and_model` (each `Dict[str, Dict[str, float]]`),
  produced by the new `compute_similarity_metrics(...)` aggregator.
  Each entry now exposes the full distribution: `mean`, `std`, `q25`,
  `median`, `q75`, `d90`, `d99`, `min`, `max`, `count`. Migration:
  replace `summary["average_by_model"][m]` (a float) with
  `summary["metrics_by_model"][m]["mean"]` (a dict value).

### Deprecated
- None.

### Removed
- None.

### Fixed
- None.

### Security
- None.
