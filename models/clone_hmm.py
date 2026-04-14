from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from models.base_hmm import BaseHMMModel
from models.base_initializer import initializer_mapping
from utils.setup_logger import get_logger

logger = get_logger("clone_hmm")


class CloneHMM(BaseHMMModel):
    """
    Clone-Structured HMM (Huang et al., 2018).
    
    Creates num_clones copies of each base state. All clones of the same
    base state share the same emission distribution, but have independent
    transitions. Total hidden states = base_states * num_clones.
    
    This allows the model to capture higher-order dependencies while
    keeping emission parameters tied (parameter-efficient).
    """
    def __init__(
        self,
        vocab_size: int,
        num_states: int,
        num_clones: int = 2,
        device: str = "cpu",
        initial_initializer: BaseHMMModel.InitializerName = "zeros",
        transition_initializer: BaseHMMModel.InitializerName = "zeros",
        emission_initializer: BaseHMMModel.InitializerName = "zeros",
    ):
        self.base_states = num_states
        self.num_clones = num_clones
        # Initialize with expanded state space: CK total states
        super().__init__(
            vocab_size=vocab_size,
            num_states=num_states * num_clones,
            initial_initializer=initial_initializer,
            transition_initializer=transition_initializer,
            emission_initializer=emission_initializer,
            device=device,
        )
        # Override emission_logits to be [base_states, V] (shared across clones)
        self.emission_logits = initializer_mapping[emission_initializer](
            (self.base_states, self.vocab_size), device=self.device
        )

    @property
    def log_emission_probs(self):
        # Tile base emissions across clones: [base_states, V] -> [CK, V]
        base_log_emit = F.log_softmax(self.emission_logits, dim=1)  # [base_states, V]
        return base_log_emit.repeat(self.num_clones, 1)  # [CK, V]

    def _calculate_emission_prob_distribution(
        self, data: list[torch.Tensor], log_posteriors: list[dict[str, torch.Tensor]]
    ) -> None:
        """
        M-step for emissions: sum gamma across all clones of the same base state.
        """
        sum_gamma_obs = torch.zeros((self.base_states, self.vocab_size), device=self.device)
        sum_gamma_total = torch.zeros((self.base_states,), device=self.device)

        for seq, post in zip(data, log_posteriors):
            gamma = torch.exp(post['log_gamma'])  # [T, CK]
            # Reshape to [T, num_clones, base_states] and sum over clones -> [T, base_states]
            gamma_by_base = gamma.reshape(-1, self.num_clones, self.base_states).sum(dim=1)
            for t, obs in enumerate(seq):
                sum_gamma_obs[:, obs] += gamma_by_base[t]
            sum_gamma_total += gamma_by_base.sum(dim=0)

        self.emission_logits.data = torch.log(sum_gamma_obs / sum_gamma_total.unsqueeze(1))
        # sanity check
        emiss_sum = torch.exp(self.emission_logits).sum(dim=1)
        assert torch.allclose(emiss_sum, torch.ones_like(emiss_sum), atol=1e-4), \
            f"Emission rows do not sum to 1: {emiss_sum}"

    def _get_metadata(self) -> dict:
        metadata = super()._get_metadata()
        metadata["num_clones"] = self.num_clones
        metadata["base_states"] = self.base_states
        return metadata

    # Everything else inherited from BaseHMMModel:
    # _forward, _backward, _compute_gamma, _compute_xi,
    # _e_step, _m_step, _sanity_check, fit, _exit_condition,
    # _calculate_initial_prob_distribution, _calculate_transition_prob_distribution,
    # log_likelihood, predict_missing, perplexity
