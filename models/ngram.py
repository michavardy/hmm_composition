from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from utils.models import Dataset, SequenceModel


class NGramModel(SequenceModel):
    def __init__(self, vocab_size: int, k: int, config: dict[str, Any] | None = None):
        super().__init__(vocab_size=vocab_size, config=config)
        self.k = int(k)
        self.discount = float(self.config.get("discount", 0.75))
        self.bos_token_id = int(self.config.get("bos_token_id", 0))
        self.order_counts: dict[int, Counter[tuple[int, ...]]] = {
            order: Counter() for order in range(1, self.k + 1)
        }
        self.context_totals: dict[int, Counter[tuple[int, ...]]] = {
            order: Counter() for order in range(2, self.k + 1)
        }
        self.context_unique_followers: dict[int, Counter[tuple[int, ...]]] = {
            order: Counter() for order in range(2, self.k + 1)
        }
        self.continuation_contexts: dict[int, set[tuple[int, ...]]] = defaultdict(set)
        self.total_continuations = 0

    def _prepare_sequence(self, sequence: np.ndarray) -> np.ndarray:
        numeric_sequence = np.asarray(sequence, dtype=np.int64)
        if numeric_sequence.size == 0:
            return numeric_sequence
        prefix = np.full(self.k - 1, self.bos_token_id, dtype=np.int64)
        return np.concatenate([prefix, numeric_sequence])

    def fit(self, data: Dataset, max_iteration: int, delta_likelyhood: float) -> NGramModel:
        del max_iteration, delta_likelyhood
        for counter in self.order_counts.values():
            counter.clear()
        for counter in self.context_totals.values():
            counter.clear()
        for counter in self.context_unique_followers.values():
            counter.clear()
        self.continuation_contexts.clear()

        for sequence in data:
            prepared = self._prepare_sequence(sequence)
            for order in range(1, self.k + 1):
                if len(prepared) < order:
                    continue
                for index in range(order - 1, len(prepared)):
                    ngram = tuple(prepared[index - order + 1 : index + 1].tolist())
                    self.order_counts[order][ngram] += 1
                    if order > 1:
                        context = ngram[:-1]
                        token = ngram[-1]
                        self.context_totals[order][context] += 1
                        self.continuation_contexts[token].add(context)

        for order in range(2, self.k + 1):
            follower_sets: dict[tuple[int, ...], set[int]] = defaultdict(set)
            for ngram in self.order_counts[order]:
                follower_sets[ngram[:-1]].add(ngram[-1])
            for context, followers in follower_sets.items():
                self.context_unique_followers[order][context] = len(followers)

        self.total_continuations = sum(len(contexts) for contexts in self.continuation_contexts.values())
        self.training_history.append(
            {
                "iteration": 1.0,
                "log_likelihood": float(sum(self.likelihood(sequence) for sequence in data)),
                "mean_log_likelihood": float(np.mean([self.likelihood(sequence) for sequence in data])) if data else 0.0,
            }
        )
        return self

    def _continuation_probability(self, token_id: int) -> float:
        if self.total_continuations == 0:
            return 1.0 / self.vocab_size
        return max(len(self.continuation_contexts[token_id]), 1) / self.total_continuations

    def _probability(self, token_id: int, history: tuple[int, ...]) -> float:
        clipped_history = history[-(self.k - 1) :]
        order = len(clipped_history) + 1
        if order <= 1:
            return self._continuation_probability(token_id)

        ngram = clipped_history + (token_id,)
        context_count = self.context_totals[order][clipped_history]
        if context_count == 0:
            return self._probability(token_id, clipped_history[1:])

        ngram_count = self.order_counts[order][ngram]
        unique_followers = self.context_unique_followers[order][clipped_history]
        discounted = max(ngram_count - self.discount, 0.0) / context_count
        lambda_weight = (self.discount * unique_followers) / context_count
        backoff = self._probability(token_id, clipped_history[1:])
        return max(discounted + lambda_weight * backoff, 1e-12)

    def likelihood(self, sequence: np.ndarray) -> float:
        numeric_sequence = np.asarray(sequence, dtype=np.int64)
        history = [self.bos_token_id] * (self.k - 1)
        log_likelihood = 0.0
        for token_id in numeric_sequence.tolist():
            probability = self._probability(int(token_id), tuple(history))
            log_likelihood += math.log(probability)
            history.append(int(token_id))
            history = history[-(self.k - 1) :]
        return float(log_likelihood)

    def predict_missing(self, sequence: np.ndarray, mask_index: int) -> int:
        numeric_sequence = np.asarray(sequence, dtype=np.int64)
        if mask_index < 0 or mask_index >= len(numeric_sequence):
            raise IndexError("mask_index is out of bounds")

        history_tokens = numeric_sequence[max(0, mask_index - self.k + 1) : mask_index].tolist()
        history = tuple(([self.bos_token_id] * (self.k - 1 - len(history_tokens))) + history_tokens)
        scores = np.array([self._probability(token_id, history) for token_id in range(self.vocab_size)])
        return int(np.argmax(scores))

    def perplexity(self, dataset: Dataset) -> float:
        total_tokens = sum(len(sequence) for sequence in dataset)
        if total_tokens == 0:
            return math.inf
        total_log_likelihood = sum(self.likelihood(sequence) for sequence in dataset)
        return float(math.exp(-total_log_likelihood / total_tokens))