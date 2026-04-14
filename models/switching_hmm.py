from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from models.base_hmm import BaseHMMModel
from utils.setup_logger import get_logger

logger = get_logger("switching_hmm")


class SwitchingHMM(BaseHMMModel):
    """
    Switching (Regime-Switching) HMM with soft mode inference.

    Defines a product state z_t = (m_t, s_t) where:
      - m_t ∈ {1,...,M} is the mode with its own transition matrix
      - s_t ∈ {1,...,K} is the state, governed by mode m_t

    Generative process per timestep:
      P(m_t | m_{t-1})       = mode transition
      P(s_t | s_{t-1}, m_t)  = state transition under current mode
      P(x_t | s_t, m_t)      = emission under current mode

    Total hidden states = M * K.  Inherits standard forward-backward
    from BaseHMMModel over the product state space.
    """

    def __init__(
        self,
        vocab_size: int,
        num_states: int,
        num_modes: int = 2,
        device: str = "cpu",
        initial_initializer: BaseHMMModel.InitializerName = "zeros",
        transition_initializer: BaseHMMModel.InitializerName = "zeros",
        emission_initializer: BaseHMMModel.InitializerName = "zeros",
    ):
        self.base_states = num_states
        self.num_modes = num_modes
        # Product state space: M * K
        super().__init__(
            vocab_size=vocab_size,
            num_states=num_modes * num_states,
            initial_initializer=initial_initializer,
            transition_initializer=transition_initializer,
            emission_initializer=emission_initializer,
            device=device,
        )

    def initialize(self, initial_initializer, transition_initializer, emission_initializer):
        """Override to create structured parameters instead of flat logits."""
        M, K = self.num_modes, self.base_states
        # Mode-level parameters
        self.mode_initial_logits = initial_initializer((M,), device=self.device)
        self.mode_transition_logits = transition_initializer((M, M), device=self.device)
        # Per-mode state-level parameters
        self.state_initial_logits = initial_initializer((M, K), device=self.device)
        self.state_transition_logits = transition_initializer((M, K, K), device=self.device)
        self.state_emission_logits = emission_initializer((M, K, self.vocab_size), device=self.device)

    @property
    def log_initial_probs(self):
        # π(m, k) = π_mode(m) * π_state(m, k)
        log_pi_mode = F.log_softmax(self.mode_initial_logits, dim=0)    # [M]
        log_pi_state = F.log_softmax(self.state_initial_logits, dim=1)  # [M, K]
        return (log_pi_mode.unsqueeze(1) + log_pi_state).reshape(-1)    # [MK]

    @property
    def log_transition_probs(self):
        # A[(m,s) → (m',s')] = A_mode(m → m') * A_state(m', s → s')
        M, K = self.num_modes, self.base_states
        log_A_mode = F.log_softmax(self.mode_transition_logits, dim=1)   # [M, M]
        log_A_state = F.log_softmax(self.state_transition_logits, dim=2) # [M_dst, K_src, K_dst]

        # Build [M_src, K_src, M_dst, K_dst] then reshape to [MK, MK]
        mode_part = log_A_mode[:, None, :, None]                         # [M, 1, M, 1]
        state_part = log_A_state.permute(1, 0, 2).unsqueeze(0)          # [1, K, M, K]
        return (mode_part + state_part).reshape(M * K, M * K)            # [MK, MK]

    @property
    def log_emission_probs(self):
        # B(m, k)(x) = B_m(x | k)
        log_B = F.log_softmax(self.state_emission_logits, dim=2)  # [M, K, V]
        return log_B.reshape(self.num_modes * self.base_states, self.vocab_size)  # [MK, V]

    # _forward, _backward, _compute_gamma, _compute_xi, _e_step, _sanity_check
    # all inherited from BaseHMMModel — they operate on the [MK] product state space

    def _calculate_initial_prob_distribution(self, log_posteriors):
        M, K = self.num_modes, self.base_states
        # Stack γ[0] across sequences: [N, MK] -> [N, M, K]
        gamma_0 = torch.stack([torch.exp(post['log_gamma'][0]) for post in log_posteriors])
        gamma_0 = gamma_0.reshape(-1, M, K)

        # Mode initial: P(m_0 = m)
        mode_counts = gamma_0.sum(dim=2).sum(dim=0)  # [M]
        self.mode_initial_logits = torch.log(mode_counts / mode_counts.sum())

        # State initial per mode: P(s_0 = k | m_0 = m)
        state_counts = gamma_0.sum(dim=0)  # [M, K]
        self.state_initial_logits = torch.log(state_counts / state_counts.sum(dim=1, keepdim=True))

    def _calculate_transition_prob_distribution(self, log_posteriors):
        M, K = self.num_modes, self.base_states
        mode_trans_counts = torch.zeros(M, M, device=self.device)
        state_trans_counts = torch.zeros(M, K, K, device=self.device)  # [M_dst, K_src, K_dst]

        for post in log_posteriors:
            xi = torch.exp(post['log_xi'])                    # [T-1, MK, MK]
            xi_4d = xi.reshape(-1, M, K, M, K)                # [T-1, M_src, K_src, M_dst, K_dst]
            mode_trans_counts += xi_4d.sum(dim=(0, 2, 4))      # [M_src, M_dst]
            # State transitions indexed by dest mode: sum over time & source mode
            state_trans_counts += xi_4d.sum(dim=(0, 1)).permute(1, 0, 2)  # [M_dst, K_src, K_dst]

        self.mode_transition_logits = torch.log(
            mode_trans_counts / mode_trans_counts.sum(dim=1, keepdim=True)
        )
        self.state_transition_logits = torch.log(
            state_trans_counts / state_trans_counts.sum(dim=2, keepdim=True)
        )

    def _calculate_emission_prob_distribution(self, data, log_posteriors):
        M, K = self.num_modes, self.base_states
        sum_gamma_obs = torch.zeros(M, K, self.vocab_size, device=self.device)
        sum_gamma_total = torch.zeros(M, K, device=self.device)

        for seq, post in zip(data, log_posteriors):
            gamma = torch.exp(post['log_gamma']).reshape(-1, M, K)  # [T, M, K]
            for t, obs in enumerate(seq):
                sum_gamma_obs[:, :, obs] += gamma[t]
            sum_gamma_total += gamma.sum(dim=0)

        self.state_emission_logits = torch.log(sum_gamma_obs / sum_gamma_total.unsqueeze(2))
        emiss_sum = torch.exp(self.state_emission_logits).sum(dim=2)
        assert torch.allclose(emiss_sum, torch.ones_like(emiss_sum), atol=1e-4), \
            f"Emission rows do not sum to 1: {emiss_sum}"

    def _get_metadata(self) -> dict:
        metadata = super()._get_metadata()
        metadata["num_modes"] = self.num_modes
        metadata["base_states"] = self.base_states
        return metadata

    # _m_step, fit, _exit_condition, log_likelihood, predict_missing, perplexity
    # all inherited from BaseHMMModel