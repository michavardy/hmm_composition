from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from utils.models import Dataset, SequenceModel
from utils.recursive import solve_recursive


def logsumexp(values: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    stable = np.exp(values - max_values)
    summed = np.log(np.sum(stable, axis=axis, keepdims=True)) + max_values
    if keepdims:
        return summed
    if axis is None:
        return np.asarray(summed).reshape(())
    return np.squeeze(summed, axis=axis)


class BaseHMMModel(SequenceModel):
    def __init__(self, vocab_size: int, num_states: int, config: dict[str, Any] | None = None):
        super().__init__(vocab_size=vocab_size, config=config)
        self.num_states = int(num_states)
        self.random_seed = int(self.config.get("random_seed", 42))
        self.pseudocount = float(self.config.get("pseudocount", 1e-3))
        self.rng = np.random.default_rng(self.random_seed)
        self.initial_probs = self._normalize(self.rng.random(self.num_states) + self.pseudocount)
        self.transition_probs = self._normalize_rows(
            self.rng.random((self.num_states, self.num_states)) + self.pseudocount
        )
        self.emission_probs = self._normalize_rows(
            self.rng.random((self.num_states, self.vocab_size)) + self.pseudocount
        )

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        total = np.sum(values)
        if total <= 0:
            return np.full_like(values, 1.0 / len(values), dtype=np.float64)
        return values / total

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return matrix / row_sums

    @property
    def log_initial_probs(self) -> np.ndarray:
        return np.log(np.clip(self.initial_probs, 1e-12, None))

    @property
    def log_transition_probs(self) -> np.ndarray:
        return np.log(np.clip(self.transition_probs, 1e-12, None))

    @property
    def log_emission_probs(self) -> np.ndarray:
        return np.log(np.clip(self.emission_probs, 1e-12, None))

    def _emission_log_table(
        self,
        sequence: np.ndarray,
        mask_index: int | None = None,
    ) -> np.ndarray:
        table = self.log_emission_probs[:, sequence].T
        if mask_index is not None:
            table[mask_index] = 0.0
        return table

    def _forward(self, emission_log_table: np.ndarray) -> np.ndarray:
        def initialize_row(_: np.ndarray, row_index: int) -> np.ndarray:
            return self.log_initial_probs + emission_log_table[row_index]

        def update_row(table: np.ndarray, row_index: int) -> np.ndarray:
            previous_row = table[row_index - 1][:, None] + self.log_transition_probs
            return logsumexp(previous_row, axis=0) + emission_log_table[row_index]

        return solve_recursive(
            len(emission_log_table),
            (self.num_states,),
            initialize_row,
            update_row,
            dtype=np.float64,
            fill_value=-np.inf,
        )

    def _backward(self, emission_log_table: np.ndarray) -> np.ndarray:
        def initialize_row(_: np.ndarray, __: int) -> np.ndarray:
            return np.zeros(self.num_states, dtype=np.float64)

        def update_row(table: np.ndarray, row_index: int) -> np.ndarray:
            next_row = emission_log_table[row_index + 1] + table[row_index + 1]
            return logsumexp(self.log_transition_probs + next_row[None, :], axis=1)

        return solve_recursive(
            len(emission_log_table),
            (self.num_states,),
            initialize_row,
            update_row,
            dtype=np.float64,
            fill_value=0.0,
            reverse=True,
        )

    def _forward_backward(
        self,
        sequence: np.ndarray,
        mask_index: int | None = None,
    ) -> dict[str, np.ndarray | float]:
        emission_log_table = self._emission_log_table(sequence, mask_index=mask_index)
        alpha = self._forward(emission_log_table)
        beta = self._backward(emission_log_table)
        log_likelihood = float(logsumexp(alpha[-1], axis=0))
        gamma = alpha + beta - log_likelihood

        xi = np.zeros((max(len(sequence) - 1, 0), self.num_states, self.num_states), dtype=np.float64)
        for time_index in range(len(sequence) - 1):
            xi[time_index] = (
                alpha[time_index][:, None]
                + self.log_transition_probs
                + emission_log_table[time_index + 1][None, :]
                + beta[time_index + 1][None, :]
                - log_likelihood
            )

        return {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "xi": xi,
            "log_likelihood": log_likelihood,
        }

    def fit(
        self,
        data: Dataset,
        max_iteration: int,
        delta_likelyhood: float,
    ) -> BaseHMMModel:
        previous_log_likelihood: float | None = None
        cleaned_data = [np.asarray(sequence, dtype=np.int64) for sequence in data if len(sequence) > 0]
        if not cleaned_data:
            raise ValueError("Training data is empty")

        for iteration in range(int(max_iteration)):
            initial_counts = np.full(self.num_states, self.pseudocount, dtype=np.float64)
            transition_counts = np.full(
                (self.num_states, self.num_states),
                self.pseudocount,
                dtype=np.float64,
            )
            emission_counts = np.full(
                (self.num_states, self.vocab_size),
                self.pseudocount,
                dtype=np.float64,
            )
            total_log_likelihood = 0.0

            for sequence in cleaned_data:
                posteriors = self._forward_backward(sequence)
                gamma = np.exp(posteriors["gamma"])
                xi = np.exp(posteriors["xi"])
                total_log_likelihood += float(posteriors["log_likelihood"])

                initial_counts += gamma[0]
                if len(sequence) > 1:
                    transition_counts += xi.sum(axis=0)
                for time_index, token_id in enumerate(sequence):
                    emission_counts[:, token_id] += gamma[time_index]

            self.initial_probs = self._normalize(initial_counts)
            self.transition_probs = self._normalize_rows(transition_counts)
            self.emission_probs = self._normalize_rows(emission_counts)
            mean_log_likelihood = total_log_likelihood / len(cleaned_data)
            self.training_history.append(
                {
                    "iteration": float(iteration + 1),
                    "log_likelihood": float(total_log_likelihood),
                    "mean_log_likelihood": float(mean_log_likelihood),
                }
            )

            if previous_log_likelihood is not None:
                improvement = abs(total_log_likelihood - previous_log_likelihood)
                if improvement <= float(delta_likelyhood):
                    break
            previous_log_likelihood = total_log_likelihood

        return self

    def likelihood(self, sequence: np.ndarray) -> float:
        result = self._forward_backward(np.asarray(sequence, dtype=np.int64))
        return float(result["log_likelihood"])

    def predict_missing(self, sequence: np.ndarray, mask_index: int) -> int:
        numeric_sequence = np.asarray(sequence, dtype=np.int64)
        if mask_index < 0 or mask_index >= len(numeric_sequence):
            raise IndexError("mask_index is out of bounds")

        posteriors = self._forward_backward(numeric_sequence, mask_index=mask_index)
        state_log_weights = posteriors["alpha"][mask_index] + posteriors["beta"][mask_index]
        state_log_weights = state_log_weights - logsumexp(state_log_weights, axis=0)
        token_probs = np.exp(state_log_weights) @ self.emission_probs
        return int(np.argmax(token_probs))

    def perplexity(self, dataset: Dataset) -> float:
        cleaned_data = [np.asarray(sequence, dtype=np.int64) for sequence in dataset if len(sequence) > 0]
        if not cleaned_data:
            return math.inf

        total_log_likelihood = float(sum(self.likelihood(sequence) for sequence in cleaned_data))
        total_tokens = sum(len(sequence) for sequence in cleaned_data)
        if total_tokens == 0:
            return math.inf
        return float(math.exp(-total_log_likelihood / total_tokens))

    def decode_states(self, sequence: np.ndarray) -> list[int]:
        numeric_sequence = np.asarray(sequence, dtype=np.int64)
        if len(numeric_sequence) == 0:
            return []

        emission_log_table = self._emission_log_table(numeric_sequence)
        backpointers = np.zeros((len(numeric_sequence), self.num_states), dtype=np.int64)

        def initialize_row(_: np.ndarray, row_index: int) -> np.ndarray:
            return self.log_initial_probs + emission_log_table[row_index]

        def update_row(table: np.ndarray, row_index: int) -> np.ndarray:
            scores = table[row_index - 1][:, None] + self.log_transition_probs
            backpointers[row_index] = np.argmax(scores, axis=0)
            return np.max(scores, axis=0) + emission_log_table[row_index]

        viterbi_table = solve_recursive(
            len(numeric_sequence),
            (self.num_states,),
            initialize_row,
            update_row,
            dtype=np.float64,
            fill_value=-np.inf,
        )
        state = int(np.argmax(viterbi_table[-1]))
        decoded = [state]
        for row_index in range(len(numeric_sequence) - 1, 0, -1):
            state = int(backpointers[row_index, state])
            decoded.append(state)
        decoded.reverse()
        return decoded


def expanded_state_count(*dimensions: int) -> int:
    count = 1
    for dimension in dimensions:
        count *= int(dimension)
    return count