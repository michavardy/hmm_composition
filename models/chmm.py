from __future__ import annotations

from typing import Any

from models.base_hmm import BaseHMMModel, expanded_state_count


class CHMMModel(BaseHMMModel):
    def __init__(self, vocab_size: int, types: int, states: int, config: dict[str, Any] | None = None):
        merged_config = dict(config or {})
        merged_config.setdefault("types", types)
        merged_config.setdefault("states", states)
        super().__init__(
            vocab_size=vocab_size,
            num_states=expanded_state_count(types, states),
            config=merged_config,
        )
        self.types = int(types)
        self.states = int(states)