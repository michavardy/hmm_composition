from __future__ import annotations

from typing import Any

from models.base_hmm import BaseHMMModel


class HMMModel(BaseHMMModel):
    def __init__(self, vocab_size: int, num_states: int):
        super().__init__(vocab_size=vocab_size, num_states=num_states)
        self.num_states = int(num_states)