from __future__ import annotations

import html
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _stringify(value) -> str:
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return f"{len(value)} rows"
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return "" if value is None else str(value)


def _load_runs(experiments_dir: Path) -> list[dict[str, str]]:
    runs: list[dict[str, str]] = []
    if not experiments_dir.exists():
        return runs

    for run_dir in sorted(experiments_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        config = _read_json(run_dir / "config.json")
        metrics = _read_json(run_dir / "metrics.json")
        row = {
            "run_id": run_dir.name,
            "name": _stringify(config.get("name", run_dir.name)),
            "model": _stringify(config.get("model_name", config.get("model", ""))),
            "train_perplexity": _stringify(metrics.get("train_perplexity")),
            "test_perplexity": _stringify(metrics.get("test_perplexity")),
            "prediction_accuracy": _stringify(metrics.get("prediction_accuracy")),
            "states": _stringify(config.get("states")),
            "max_iteration": _stringify(config.get("max_iteration")),
            "train_subset": _stringify(config.get("train_subset")),
            "test_subset": _stringify(config.get("test_subset")),
        }
        runs.append(row)

    runs.sort(key=lambda item: item["run_id"], reverse=True)
    return runs


def build_table_html(experiments_dir: Path, table_template_path: Path) -> str:
    columns = [
        ("run_id", "Run"),
        ("name", "Name"),
        ("model", "Model"),
        ("train_perplexity", "Train Perplexity"),
        ("test_perplexity", "Test Perplexity"),
        ("prediction_accuracy", "Prediction Accuracy"),
        ("states", "States"),
        ("max_iteration", "Max Iter"),
        ("train_subset", "Train Subset"),
        ("test_subset", "Test Subset"),
    ]
    runs = _load_runs(experiments_dir)

    header_html = "".join(
        f"<th>{html.escape(label)}</th>" for _, label in columns
    )
    if runs:
        rows_html = "".join(
            "<tr>"
            + "".join(
                f"<td>{html.escape(row.get(key, ''))}</td>" for key, _ in columns
            )
            + "</tr>"
            for row in runs
        )
    else:
        rows_html = (
            f"<tr><td colspan=\"{len(columns)}\">No experiment runs found in "
            f"{html.escape(str(experiments_dir))}</td></tr>"
        )

    template = table_template_path.read_text(encoding="utf-8")
    return (
        template.replace("{{HEADER_ROW}}", header_html)
        .replace("{{BODY_ROWS}}", rows_html)
        .replace("{{RUN_COUNT}}", str(len(runs)))
        .replace("{{EXPERIMENTS_DIR}}", html.escape(str(experiments_dir)))
    )