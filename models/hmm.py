from __future__ import annotations

from typing import Any

from models.base_hmm import BaseHMMModel


class HMMModel(BaseHMMModel):
    def __init__(self, vocab_size: int, states: int, config: dict[str, Any] | None = None):
        super().__init__(vocab_size=vocab_size, num_states=states, config=config)
        self.states = int(states)