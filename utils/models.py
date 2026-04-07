from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


ArraySequence = np.ndarray
Dataset = list[ArraySequence]


class SequenceModel(ABC):
    def __init__(self, vocab_size: int, config: dict[str, Any] | None = None):
        self.vocab_size = int(vocab_size)
        self.config = dict(config or {})
        self.training_history: list[dict[str, float]] = []

    @abstractmethod
    def fit(
        self,
        data: Dataset,
        max_iteration: int,
        delta_likelyhood: float,
    ) -> SequenceModel:
        raise NotImplementedError

    @abstractmethod
    def likelihood(self, sequence: ArraySequence) -> float:
        raise NotImplementedError

    @abstractmethod
    def predict_missing(self, sequence: ArraySequence, mask_index: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def perplexity(self, dataset: Dataset) -> float:
        raise NotImplementedError

    def dump(self, output_path: str | Path) -> Path:
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as output_file:
            pickle.dump(self, output_file, protocol=pickle.HIGHEST_PROTOCOL)
        return target_path

    @classmethod
    def load(cls, input_path: str | Path) -> SequenceModel:
        with Path(input_path).open("rb") as input_file:
            model = pickle.load(input_file)
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(model).__name__}")
        return model


@dataclass(slots=True)
class TrainSummary:
    train_log_likelihood: float
    train_perplexity: float
    test_log_likelihood: float | None = None
    test_perplexity: float | None = None
    history: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_log_likelihood": self.train_log_likelihood,
            "train_perplexity": self.train_perplexity,
            "test_log_likelihood": self.test_log_likelihood,
            "test_perplexity": self.test_perplexity,
            "history": self.history,
        }


class Trainer:
    def __init__(
        self,
        model: SequenceModel,
        train_data: Dataset,
        val_data: Dataset | None = None,
    ):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data

    def train(self, max_iter: int = 30, delta_likelyhood: float = 1e-3) -> TrainSummary:
        self.model.fit(
            self.train_data,
            max_iteration=max_iter,
            delta_likelyhood=delta_likelyhood,
        )
        train_metrics = self.evaluate(self.train_data)
        val_metrics = self.evaluate(self.val_data) if self.val_data is not None else None
        return TrainSummary(
            train_log_likelihood=train_metrics["mean_log_likelihood"],
            train_perplexity=train_metrics["perplexity"],
            test_log_likelihood=None if val_metrics is None else val_metrics["mean_log_likelihood"],
            test_perplexity=None if val_metrics is None else val_metrics["perplexity"],
            history=list(self.model.training_history),
        )

    def evaluate(self, dataset: Dataset | None) -> dict[str, float] | None:
        if not dataset:
            return None

        total_log_likelihood = float(sum(self.model.likelihood(sequence) for sequence in dataset))
        return {
            "mean_log_likelihood": total_log_likelihood / len(dataset),
            "perplexity": self.model.perplexity(dataset),
        }


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as input_file:
        return pickle.load(input_file)


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target_path