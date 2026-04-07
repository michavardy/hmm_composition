from __future__ import annotations

from typing import Any

from models.base_hmm import BaseHMMModel, expanded_state_count


class FiniteHMMModel(BaseHMMModel):
    def __init__(
        self,
        vocab_size: int,
        memory_window: int,
        states: int,
        config: dict[str, Any] | None = None,
    ):
        merged_config = dict(config or {})
        merged_config.setdefault("memory_window", memory_window)
        merged_config.setdefault("states", states)
        super().__init__(
            vocab_size=vocab_size,
            num_states=expanded_state_count(*([states] * (int(memory_window) + 1))),
            config=merged_config,
        )
        self.memory_window = int(memory_window)
        self.states = int(states)