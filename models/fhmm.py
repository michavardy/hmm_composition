from __future__ import annotations

from typing import Any

from models.base_hmm import BaseHMMModel, expanded_state_count


class FHMMModel(BaseHMMModel):
    def __init__(self, vocab_size: int, chains: int, states: int, config: dict[str, Any] | None = None):
        merged_config = dict(config or {})
        merged_config.setdefault("chains", chains)
        merged_config.setdefault("states", states)
        super().__init__(
            vocab_size=vocab_size,
            num_states=expanded_state_count(*([states] * int(chains))),
            config=merged_config,
        )
        self.chains = int(chains)
        self.states = int(states)