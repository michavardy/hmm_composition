from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models import MODEL_REGISTRY
from utils.models import Dataset, Trainer, load_pickle, save_json
from utils.setup_logger import get_logger

try:
    import wandb
except ImportError:  # pragma: no cover - dependency guard
    wandb = None


logger = get_logger("TRAIN")
CONFIG_PATH = ROOT_DIR / "utils" / "config.toml"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("rb") as config_file:
        return tomllib.load(config_file)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def load_processed_artifacts(config: dict[str, Any]) -> tuple[Dataset, Dataset, dict[str, Any]]:
    processed_dir = resolve_path(config["processed_dir"])
    train_data = load_pickle(processed_dir / config["train_data_file"])
    test_data = load_pickle(processed_dir / config["test_data_file"])
    vocab = load_pickle(processed_dir / config["vocab_file"])
    return train_data, test_data, vocab


def next_run_directory(experiments_dir: Path) -> Path:
    experiments_dir.mkdir(parents=True, exist_ok=True)
    existing_runs = []
    for child in experiments_dir.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"run_(\d{3})", child.name)
        if match:
            existing_runs.append(int(match.group(1)))
    next_index = max(existing_runs, default=0) + 1
    run_dir = experiments_dir / f"run_{next_index:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


def limit_dataset(dataset: Dataset, limit: int) -> Dataset:
    if limit <= 0 or limit >= len(dataset):
        return dataset
    return dataset[:limit]


def model_section_name(model_name: str) -> str:
    if model_name == "n_gram":
        return "model_n_gram"
    if model_name == "f_hmm":
        return "model_f_hmm"
    return f"model_{model_name}"


def merged_experiment_config(config: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    model_name = experiment["model"]
    merged = {key: value for key, value in config["training"].items() if key != "experiments"}
    merged.update(config.get(model_section_name(model_name), {}))
    merged.update(experiment)
    merged["model_name"] = model_name
    return merged


def build_model(model_name: str, vocab_size: int, experiment_config: dict[str, Any]):
    model_cls = MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        raise ValueError(f"Unsupported model '{model_name}'")

    constructor_args: dict[str, Any] = {"vocab_size": vocab_size, "config": experiment_config}
    if model_name == "hmm":
        constructor_args["states"] = int(experiment_config["states"])
    elif model_name == "chmm":
        constructor_args["types"] = int(experiment_config["types"])
        constructor_args["states"] = int(experiment_config["states"])
    elif model_name == "fhmm":
        constructor_args["chains"] = int(experiment_config["chains"])
        constructor_args["states"] = int(experiment_config["states"])
    elif model_name == "f_hmm":
        constructor_args["memory_window"] = int(experiment_config["memory_window"])
        constructor_args["states"] = int(experiment_config["states"])
    elif model_name == "switching_hmm":
        constructor_args["modes"] = int(experiment_config["modes"])
        constructor_args["states"] = int(experiment_config["states"])
    elif model_name == "lstm":
        constructor_args["layers"] = int(experiment_config["layers"])
        constructor_args["nodes"] = int(experiment_config["nodes"])
    elif model_name == "n_gram":
        constructor_args["k"] = int(experiment_config["k"])
    return model_cls(**constructor_args)


def evaluate_prediction_accuracy(model, dataset: Dataset) -> float | None:
    if not dataset:
        return None

    correct = 0
    total = 0
    for sequence in dataset:
        if len(sequence) < 2:
            continue
        mask_index = len(sequence) // 2
        predicted = model.predict_missing(sequence, mask_index)
        if int(predicted) == int(sequence[mask_index]):
            correct += 1
        total += 1
    if total == 0:
        return None
    return correct / total


def start_wandb(experiment_config: dict[str, Any], run_dir: Path):
    if wandb is None:
        logger.warning("wandb is not installed; skipping experiment tracking")
        return None

    run = wandb.init(
        project=experiment_config["wandb_project"],
        entity=experiment_config.get("wandb_entity") or None,
        mode=experiment_config.get("wandb_mode", "offline"),
        dir=str(run_dir / "logs"),
        name=experiment_config.get("name"),
        tags=list(experiment_config.get("tags", [])),
        config=experiment_config,
        reinit="finish_previous",
    )
    return run


def run_experiment(
    train_data: Dataset,
    test_data: Dataset,
    vocab: dict[str, Any],
    config: dict[str, Any],
    experiment: dict[str, Any],
) -> dict[str, Any]:
    experiment_config = merged_experiment_config(config, experiment)
    run_dir = next_run_directory(resolve_path(config["training"]["experiments_dir"]))
    save_json(run_dir / "config.json", experiment_config)

    limited_train = limit_dataset(train_data, int(experiment_config.get("train_subset", 0)))
    limited_test = limit_dataset(test_data, int(experiment_config.get("test_subset", 0)))
    vocab_size = len(vocab["word_to_id"])
    model = build_model(experiment_config["model_name"], vocab_size, experiment_config)

    logger.info(
        "Running experiment %s with model=%s train=%s test=%s",
        experiment_config["name"],
        experiment_config["model_name"],
        len(limited_train),
        len(limited_test),
    )

    wandb_run = start_wandb(experiment_config, run_dir)
    trainer = Trainer(model, limited_train, limited_test)
    summary = trainer.train(
        max_iter=int(experiment_config["max_iteration"]),
        delta_likelyhood=float(experiment_config["delta_likelyhood"]),
    )
    metrics = summary.to_dict()
    metrics["prediction_accuracy"] = evaluate_prediction_accuracy(model, limited_test)
    metrics["train_sequences"] = len(limited_train)
    metrics["test_sequences"] = len(limited_test)
    metrics["vocab_size"] = vocab_size

    model.dump(run_dir / "model.pkl")
    save_json(run_dir / "metrics.json", metrics)

    if wandb_run is not None:
        if summary.history:
            for step, history_row in enumerate(summary.history):
                wandb_run.log({f"history/{key}": value for key, value in history_row.items()}, step=step)
        wandb_run.log(
            {key: value for key, value in metrics.items() if key != "history"},
            step=len(summary.history),
        )
        wandb_run.finish()

    return {
        "run_dir": str(run_dir),
        "experiment": experiment_config["name"],
        "model": experiment_config["model_name"],
        "metrics": metrics,
    }


def run_training() -> list[dict[str, Any]]:
    config = load_config()
    training_config = config.get("training", {})
    experiments = [exp for exp in training_config.get("experiments", []) if exp.get("enabled", True)]
    if not experiments:
        raise ValueError("No enabled experiments found in [training].")

    train_data, test_data, vocab = load_processed_artifacts(training_config)
    results = []
    for experiment in experiments:
        results.append(run_experiment(train_data, test_data, vocab, config, experiment))

    return results


if __name__ == "__main__":
    experiment_results = run_training()
    print(json.dumps(experiment_results, indent=2))