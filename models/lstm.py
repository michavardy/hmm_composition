from __future__ import annotations

import math
from typing import Any

import numpy as np

from utils.models import Dataset, SequenceModel

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - dependency guard
    torch = None
    nn = None


if nn is not None:
    class _TokenLSTM(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            hidden_size: int,
            layers: int,
            dropout: float,
        ):
            super().__init__()
            effective_dropout = dropout if layers > 1 else 0.0
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=layers,
                dropout=effective_dropout,
                batch_first=True,
            )
            self.output = nn.Linear(hidden_size, vocab_size)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            embedded = self.embedding(inputs)
            outputs, _ = self.lstm(embedded)
            return self.output(outputs)
else:
    class _TokenLSTM:  # pragma: no cover - dependency guard
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required to use LSTMModel")


class LSTMModel(SequenceModel):
    def __init__(self, vocab_size: int, layers: int, nodes: int, config: dict[str, Any] | None = None):
        if torch is None or nn is None:
            raise ImportError("torch is required to use LSTMModel")

        super().__init__(vocab_size=vocab_size, config=config)
        self.layers = int(layers)
        self.nodes = int(nodes)
        self.embedding_dim = int(self.config.get("embedding_dim", nodes))
        self.learning_rate = float(self.config.get("learning_rate", 1e-3))
        self.epochs = int(self.config.get("epochs", 1))
        self.dropout = float(self.config.get("dropout", 0.0))
        self.bos_token_id = int(self.config.get("bos_token_id", 0))
        self.device = torch.device(self.config.get("device", "cpu"))
        self.network = _TokenLSTM(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_size=self.nodes,
            layers=self.layers,
            dropout=self.dropout,
        ).to(self.device)

    def _tensorize(self, sequence: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        numeric_sequence = np.asarray(sequence, dtype=np.int64)
        if len(numeric_sequence) == 0:
            raise ValueError("Empty sequence cannot be used for training")

        inputs = np.concatenate([[self.bos_token_id], numeric_sequence[:-1]])
        inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=self.device).unsqueeze(0)
        targets_tensor = torch.tensor(numeric_sequence, dtype=torch.long, device=self.device).unsqueeze(0)
        return inputs_tensor, targets_tensor

    def fit(self, data: Dataset, max_iteration: int, delta_likelyhood: float) -> LSTMModel:
        del max_iteration, delta_likelyhood
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        sequences = [np.asarray(sequence, dtype=np.int64) for sequence in data if len(sequence) > 0]

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            token_count = 0
            self.network.train()
            for sequence in sequences:
                inputs, targets = self._tensorize(sequence)
                optimizer.zero_grad()
                logits = self.network(inputs)
                loss = criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(sequence)
                token_count += len(sequence)

            mean_loss = epoch_loss / max(token_count, 1)
            self.training_history.append(
                {
                    "iteration": float(epoch + 1),
                    "log_likelihood": float(-epoch_loss),
                    "mean_log_likelihood": float(-mean_loss),
                }
            )

        return self

    def _logits(self, sequence: np.ndarray) -> torch.Tensor:
        inputs, _ = self._tensorize(sequence)
        self.network.eval()
        with torch.no_grad():
            return self.network(inputs).squeeze(0)

    def likelihood(self, sequence: np.ndarray) -> float:
        numeric_sequence = np.asarray(sequence, dtype=np.int64)
        if len(numeric_sequence) == 0:
            return 0.0

        logits = self._logits(numeric_sequence)
        log_probs = torch.log_softmax(logits, dim=-1)
        positions = torch.arange(len(numeric_sequence), device=self.device)
        targets = torch.tensor(numeric_sequence, dtype=torch.long, device=self.device)
        selected = log_probs[positions, targets]
        return float(selected.sum().cpu().item())

    def predict_missing(self, sequence: np.ndarray, mask_index: int) -> int:
        numeric_sequence = np.asarray(sequence, dtype=np.int64)
        if mask_index < 0 or mask_index >= len(numeric_sequence):
            raise IndexError("mask_index is out of bounds")

        prefix = numeric_sequence[:mask_index]
        inputs = np.concatenate([[self.bos_token_id], prefix]) if len(prefix) else np.array([self.bos_token_id])
        inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=self.device).unsqueeze(0)
        self.network.eval()
        with torch.no_grad():
            logits = self.network(inputs_tensor).squeeze(0)[-1]
        return int(torch.argmax(logits).cpu().item())

    def perplexity(self, dataset: Dataset) -> float:
        total_tokens = sum(len(sequence) for sequence in dataset)
        if total_tokens == 0:
            return math.inf
        total_log_likelihood = sum(self.likelihood(sequence) for sequence in dataset)
        return float(math.exp(-total_log_likelihood / total_tokens))