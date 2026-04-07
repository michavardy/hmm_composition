from __future__ import annotations

from typing import Any

from models.base_hmm import BaseHMMModel, expanded_state_count


class SwitchingHMMModel(BaseHMMModel):
    def __init__(self, vocab_size: int, modes: int, states: int, config: dict[str, Any] | None = None):
        merged_config = dict(config or {})
        merged_config.setdefault("modes", modes)
        merged_config.setdefault("states", states)
        super().__init__(
            vocab_size=vocab_size,
            num_states=expanded_state_count(modes, states),
            config=merged_config,
        )
        self.modes = int(modes)
        self.states = int(states)