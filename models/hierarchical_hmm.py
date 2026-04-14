from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from models.base_hmm import BaseHMMModel
from models.base_initializer import initializer_mapping
from utils.setup_logger import get_logger

logger = get_logger("hierarchical_hmm")


class HierarchicalHMM(BaseHMMModel):
    """
    2-level Hierarchical HMM (Fine, Singer & Tishby, 1998).

    Top level: D production states, each activating a sub-HMM.
    Bottom level: each sub-HMM has K internal states that emit observations.

    Transition structure:
      - Each sub-state (d, k) has an end probability p_end(d, k).
      - If NOT ending: horizontal transition within the same sub-HMM.
        A[(d,k) → (d,k')] = (1 - p_end[d,k]) * A_sub[d, k, k']
      - If ending: return to parent, parent transitions, new child activated.
        A[(d,k) → (d',k')] = p_end[d,k] * A_top[d, d'] * π_sub[d', k']

    Total hidden states = D * K.
    """

    def __init__(
        self,
        vocab_size: int,
        num_states: int,
        num_productions: int = 3,
        device: str = "cpu",
        initial_initializer: BaseHMMModel.InitializerName = "zeros",
        transition_initializer: BaseHMMModel.InitializerName = "zeros",
        emission_initializer: BaseHMMModel.InitializerName = "zeros",
    ):
        self.num_productions = num_productions  # D (top-level states)
        self.sub_states = num_states             # K (states per sub-HMM)
        super().__init__(
            vocab_size=vocab_size,
            num_states=num_productions * num_states,
            initial_initializer=initial_initializer,
            transition_initializer=transition_initializer,
            emission_initializer=emission_initializer,
            device=device,
        )

    def initialize(self, initial_initializer, transition_initializer, emission_initializer):
        D, K = self.num_productions, self.sub_states
        # Top-level parameters
        self.top_initial_logits = initial_initializer((D,), device=self.device)
        self.top_transition_logits = transition_initializer((D, D), device=self.device)
        # Per-sub-HMM parameters
        self.sub_initial_logits = initial_initializer((D, K), device=self.device)
        self.sub_transition_logits = transition_initializer((D, K, K), device=self.device)
        # End probability per sub-state (logit for sigmoid)
        self.sub_end_logits = torch.full((D, K), -2.0, device=self.device)  # ~0.12 initial end prob
        # Emissions per product state
        self.emission_logits = emission_initializer((D, K, self.vocab_size), device=self.device)

    @property
    def log_initial_probs(self):
        # π(d, k) = π_top(d) * π_sub(d, k)
        log_pi_top = F.log_softmax(self.top_initial_logits, dim=0)     # [D]
        log_pi_sub = F.log_softmax(self.sub_initial_logits, dim=1)     # [D, K]
        return (log_pi_top.unsqueeze(1) + log_pi_sub).reshape(-1)       # [DK]

    @property
    def log_transition_probs(self):
        D, K = self.num_productions, self.sub_states

        # End probabilities
        log_p_end = F.logsigmoid(self.sub_end_logits)                    # [D, K]
        log_p_stay = F.logsigmoid(-self.sub_end_logits)                  # [D, K] = log(1 - p_end)

        # Sub-HMM transitions (within same production)
        log_A_sub = F.log_softmax(self.sub_transition_logits, dim=2)     # [D, K_src, K_dst]

        # Top-level transitions (on return)
        log_A_top = F.log_softmax(self.top_transition_logits, dim=1)     # [D_src, D_dst]

        # Sub-HMM initial (on activation of new child)
        log_pi_sub = F.log_softmax(self.sub_initial_logits, dim=1)       # [D, K]

        # Build full [DK, DK] transition matrix
        log_trans = torch.full((D * K, D * K), -float('inf'), device=self.device)

        for d in range(D):
            for k in range(K):
                src = d * K + k

                # Stay path: (d,k) → (d,k') with prob (1-p_end) * A_sub[d,k,k']
                for k2 in range(K):
                    dst = d * K + k2
                    log_trans[src, dst] = log_p_stay[d, k] + log_A_sub[d, k, k2]

                # Return path: (d,k) → (d',k') with prob p_end * A_top[d,d'] * π_sub[d',k']
                for d2 in range(D):
                    for k2 in range(K):
                        dst = d2 * K + k2
                        return_val = log_p_end[d, k] + log_A_top[d, d2] + log_pi_sub[d2, k2]
                        # logsumexp to combine stay and return paths to the same destination
                        log_trans[src, dst] = torch.logaddexp(log_trans[src, dst], return_val)

        return log_trans  # [DK, DK]

    @property
    def log_emission_probs(self):
        log_B = F.log_softmax(self.emission_logits, dim=2)  # [D, K, V]
        return log_B.reshape(self.num_productions * self.sub_states, self.vocab_size)  # [DK, V]

    # _forward, _backward, _compute_gamma, _compute_xi, _e_step, _sanity_check
    # all inherited from BaseHMMModel operating on [DK] product state space

    def _calculate_initial_prob_distribution(self, log_posteriors):
        D, K = self.num_productions, self.sub_states
        gamma_0 = torch.stack([torch.exp(post['log_gamma'][0]) for post in log_posteriors])  # [N, DK]
        gamma_0 = gamma_0.reshape(-1, D, K)  # [N, D, K]

        # Top-level initial
        top_counts = gamma_0.sum(dim=2).sum(dim=0)  # [D]
        self.top_initial_logits = torch.log(top_counts / top_counts.sum())

        # Sub-HMM initial per production
        sub_counts = gamma_0.sum(dim=0)  # [D, K]
        self.sub_initial_logits = torch.log(sub_counts / sub_counts.sum(dim=1, keepdim=True))

    def _calculate_transition_prob_distribution(self, log_posteriors):
        D, K = self.num_productions, self.sub_states

        # Accumulate in product space, then decompose
        top_trans_counts = torch.zeros(D, D, device=self.device)
        sub_trans_counts = torch.zeros(D, K, K, device=self.device)
        end_counts = torch.zeros(D, K, device=self.device)
        stay_counts = torch.zeros(D, K, device=self.device)

        for post in log_posteriors:
            xi = torch.exp(post['log_xi'])  # [T-1, DK, DK]
            xi_4d = xi.reshape(-1, D, K, D, K)  # [T-1, D_src, K_src, D_dst, K_dst]

            # Within-production transitions (d_src == d_dst): stay events
            for d in range(D):
                sub_trans_counts[d] += xi_4d[:, d, :, d, :].sum(dim=0)  # [K, K]

            # Cross-production transitions (d_src != d_dst): return events
            xi_time_sum = xi_4d.sum(dim=0)  # [D_src, K_src, D_dst, K_dst]
            for d_src in range(D):
                for k_src in range(K):
                    within = xi_time_sum[d_src, k_src, d_src, :].sum()
                    total = xi_time_sum[d_src, k_src, :, :].sum()
                    cross = total - within
                    stay_counts[d_src, k_src] += within
                    end_counts[d_src, k_src] += cross

            # Top-level transitions: marginalize over sub-states
            top_trans_counts += xi_time_sum.sum(dim=(1, 3))  # [D_src, D_dst]

        # Update top-level transitions
        eps = 1e-8
        self.top_transition_logits = torch.log(
            top_trans_counts / (top_trans_counts.sum(dim=1, keepdim=True) + eps)
        )

        # Update sub-HMM transitions (within-production only)
        self.sub_transition_logits = torch.log(
            sub_trans_counts / (sub_trans_counts.sum(dim=2, keepdim=True) + eps)
        )

        # Update end probabilities
        total_counts = end_counts + stay_counts + eps
        self.sub_end_logits = torch.log(end_counts / total_counts) - torch.log(stay_counts / total_counts)
        # logit = log(p_end / (1 - p_end))

    def _calculate_emission_prob_distribution(self, data, log_posteriors):
        D, K = self.num_productions, self.sub_states
        sum_gamma_obs = torch.zeros(D, K, self.vocab_size, device=self.device)
        sum_gamma_total = torch.zeros(D, K, device=self.device)

        for seq, post in zip(data, log_posteriors):
            gamma = torch.exp(post['log_gamma']).reshape(-1, D, K)  # [T, D, K]
            for t, obs in enumerate(seq):
                sum_gamma_obs[:, :, obs] += gamma[t]
            sum_gamma_total += gamma.sum(dim=0)

        self.emission_logits = torch.log(sum_gamma_obs / sum_gamma_total.unsqueeze(2))
        emiss_sum = torch.exp(self.emission_logits).sum(dim=2)
        assert torch.allclose(emiss_sum, torch.ones_like(emiss_sum), atol=1e-4), \
            f"Emission rows do not sum to 1: {emiss_sum}"

    def _get_metadata(self) -> dict:
        metadata = super()._get_metadata()
        metadata["num_productions"] = self.num_productions
        metadata["sub_states"] = self.sub_states
        return metadata

    # _m_step, fit, _exit_condition, log_likelihood, predict_missing, perplexity
    # all inherited from BaseHMMModel
