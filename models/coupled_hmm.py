from __future__ import annotations

from typing import Literal

import json
import pickle
import random
import string
import time
from pathlib import Path

import numpy as np
import torch
from models.base_hmm import BaseHMMModel
from utils.constants import trained_model_path, trained_model_metadata_path
from utils.setup_logger import get_logger

logger = get_logger("coupled_hmm")


class CoupledHMM:
    """
    Coupled HMM (Brand et al., 1997).

    Multiple chains where each chain's TRANSITIONS depend on the other
    chains' states. Contrast with factorial HMM which couples through emissions.

    For chain c, the transition at time t is:
      P(s_t^c | s_{t-1}^c, {s_{t-1}^{c'}}_{c'!=c})
    approximated via mean-field as:
      A_eff^c[i,j] = A_base^c[i,j] + sum_{c'!=c} sum_k gamma^{c'}_{t-1,k} * W^{c,c'}[k,j]

    where W^{c,c'}[k,j] is a coupling matrix: how state k in chain c'
    influences the transition into state j in chain c.

    Uses mean-field coordinate ascent like factorial HMM, but the
    coupling is in transitions rather than emissions.
    """

    def __init__(
        self,
        vocab_size: int,
        num_chains: int,
        num_states: int,
        device: str = "cpu",
        initial_initializer: BaseHMMModel.InitializerName = "zeros",
        transition_initializer: BaseHMMModel.InitializerName = "zeros",
        emission_initializer: BaseHMMModel.InitializerName = "zeros",
    ):
        self.vocab_size = vocab_size
        self.num_chains = num_chains
        self.num_states = num_states
        self.device = device
        self.initial_initializer_name = initial_initializer
        self.transition_initializer_name = transition_initializer
        self.emission_initializer_name = emission_initializer
        self.iterations_run = 0
        self.final_log_likelihood = None
        self.fit_time = None

        # One BaseHMM per chain (owns its own initial, transition, emission)
        self.chain_hmms: list[BaseHMMModel] = []
        for _ in range(num_chains):
            self.chain_hmms.append(
                BaseHMMModel(
                    vocab_size=vocab_size,
                    num_states=num_states,
                    device=device,
                    initial_initializer=initial_initializer,
                    transition_initializer=transition_initializer,
                    emission_initializer=emission_initializer,
                )
            )

        # Coupling matrices: W[c_src, c_dst, K_src, K_dst]
        # How chain c_src's state influences chain c_dst's transitions
        self.coupling_logits = torch.zeros(
            num_chains, num_chains, num_states, num_states, device=device
        ) * 0.01  # small initial coupling

    def get_cleaned_data(self, data: list[np.array]) -> list[torch.Tensor]:
        return [
            seq.detach().clone().to(dtype=torch.long, device=self.device)
            if isinstance(seq, torch.Tensor)
            else torch.tensor(seq, dtype=torch.long, device=self.device)
            for seq in data
        ]

    def _get_effective_transition(self, target_idx: int, chain_gammas: dict[int, torch.Tensor], t: int) -> torch.Tensor:
        """
        Compute effective log-transition for chain target_idx at time t,
        incorporating coupling from other chains' posteriors at t-1.

        Returns: [K, K] effective log-transition matrix for this timestep.
        """
        hmm = self.chain_hmms[target_idx]
        log_A = hmm.log_transition_probs  # [K, K]

        # Add coupling contribution from each other chain
        coupling_contrib = torch.zeros(self.num_states, device=self.device)  # [K_dst]
        for ci in range(self.num_chains):
            if ci == target_idx:
                continue
            gamma_prev = torch.exp(chain_gammas[ci][t - 1])  # [K]
            # W[ci, target_idx] is [K_src, K_dst]
            # Expected coupling: sum_k gamma[k] * W[k, j] -> [K_dst]
            coupling_contrib += gamma_prev @ self.coupling_logits[ci, target_idx]

        # Add coupling as a bias to the log-transition (then re-normalize per row)
        biased = log_A + coupling_contrib.unsqueeze(0)  # [K, K]
        # Re-normalize rows in log-space
        return biased - torch.logsumexp(biased, dim=1, keepdim=True)  # [K, K]

    def _forward_coupled(self, sequence: torch.Tensor, target_idx: int,
                         chain_gammas: dict[int, torch.Tensor]) -> torch.Tensor:
        """Forward pass with time-varying effective transitions."""
        T = len(sequence)
        K = self.num_states
        hmm = self.chain_hmms[target_idx]
        alpha = torch.zeros((T, K), device=self.device)  # [T, K]
        alpha[0] = hmm.log_initial_probs + hmm.log_emission_probs[:, sequence[0]]  # [K]

        for t in range(1, T):
            eff_trans = self._get_effective_transition(target_idx, chain_gammas, t)  # [K, K]
            scores = alpha[t - 1].unsqueeze(1) + eff_trans  # [K,1] + [K,K] -> [K,K]
            alpha[t] = torch.logsumexp(scores, dim=0) + hmm.log_emission_probs[:, sequence[t]]  # [K]
        return alpha  # [T, K]

    def _backward_coupled(self, sequence: torch.Tensor, target_idx: int,
                          chain_gammas: dict[int, torch.Tensor]) -> torch.Tensor:
        """Backward pass with time-varying effective transitions."""
        T = len(sequence)
        K = self.num_states
        hmm = self.chain_hmms[target_idx]
        beta = torch.zeros((T, K), device=self.device)  # [T, K]

        for t in range(T - 2, -1, -1):
            eff_trans = self._get_effective_transition(target_idx, chain_gammas, t + 1)  # [K, K]
            scores = eff_trans + hmm.log_emission_probs[:, sequence[t + 1]] + beta[t + 1]  # [K,K]+[K]+[K]
            beta[t] = torch.logsumexp(scores, dim=1)  # [K]
        return beta  # [T, K]

    def _compute_gamma(self, log_alpha: torch.Tensor, log_beta: torch.Tensor) -> torch.Tensor:
        log_gamma = log_alpha + log_beta  # [T, K]
        log_gamma -= torch.logsumexp(log_gamma, dim=1, keepdim=True)
        return log_gamma  # [T, K]

    def _compute_xi_coupled(self, sequence: torch.Tensor, log_alpha: torch.Tensor,
                            log_beta: torch.Tensor, target_idx: int,
                            chain_gammas: dict[int, torch.Tensor]) -> torch.Tensor:
        """Xi with time-varying effective transitions."""
        T = len(sequence)
        K = self.num_states
        hmm = self.chain_hmms[target_idx]
        log_xi = torch.zeros((T - 1, K, K), device=self.device)  # [T-1, K, K]

        for t in range(T - 1):
            eff_trans = self._get_effective_transition(target_idx, chain_gammas, t + 1)
            scores = (
                log_alpha[t].unsqueeze(1)                           # [K, 1]
                + eff_trans                                          # [K, K]
                + hmm.log_emission_probs[:, sequence[t + 1]].unsqueeze(0)  # [1, K]
                + log_beta[t + 1]                                    # [K]
            )  # [K, K]
            log_norm = torch.logsumexp(scores.flatten(), dim=0)
            log_xi[t] = scores - log_norm

        return log_xi  # [T-1, K, K]

    @torch.no_grad()
    def _e_step(self, data: list[torch.Tensor], num_mf_iterations: int = 5,
                mf_tol: float = 1e-4) -> list[dict[str, torch.Tensor]]:
        """
        E-step with mean-field coordinate ascent.
        Same structure as factorial HMM but coupling is in transitions.
        """
        K = self.num_states

        chain_posteriors = [
            {'chain_idx': i, 'hmm': self.chain_hmms[i], 'posteriors': []}
            for i in range(self.num_chains)
        ]

        for sequence in data:  # sequence: [T]
            T = len(sequence)
            chain_gammas = {i: torch.full((T, K), -np.log(K), device=self.device)
                           for i in range(self.num_chains)}
            seq_data = {}

            for mf_iter in range(num_mf_iterations):
                max_delta = 0.0
                for ci in range(self.num_chains):
                    alpha = self._forward_coupled(sequence, ci, chain_gammas)
                    beta = self._backward_coupled(sequence, ci, chain_gammas)
                    gamma = self._compute_gamma(alpha, beta)

                    delta = (torch.exp(gamma) - torch.exp(chain_gammas[ci])).abs().sum().item()
                    max_delta = max(max_delta, delta)
                    chain_gammas[ci] = gamma
                    seq_data[ci] = (alpha, beta)

                if max_delta < mf_tol:
                    logger.debug(f"MF converged at iter {mf_iter} (delta={max_delta:.6f})")
                    break

            # Compute xi once per chain using converged posteriors
            for ci in range(self.num_chains):
                a, b = seq_data[ci]
                chain_posteriors[ci]['posteriors'].append({
                    'log_alpha': a,
                    'log_beta': b,
                    'log_gamma': chain_gammas[ci],
                    'log_xi': self._compute_xi_coupled(sequence, a, b, ci, chain_gammas),
                })

        return chain_posteriors

    def _m_step(self, data: list[torch.Tensor], chain_posteriors: list[dict]) -> None:
        """Update each chain's base parameters and the coupling matrices."""
        # Update base HMM parameters per chain (reuse BaseHMMModel._m_step)
        for chain_post in chain_posteriors:
            hmm = chain_post['hmm']
            hmm._m_step(data, chain_post['posteriors'])

        # Update coupling matrices from xi residuals
        # For each (c_src, c_dst) pair, W[c_src, c_dst, k, j] captures
        # how much chain c_src being in state k biases chain c_dst toward state j
        for c_dst in range(self.num_chains):
            posteriors = chain_posteriors[c_dst]['posteriors']
            for c_src in range(self.num_chains):
                if c_src == c_dst:
                    continue
                src_posteriors = chain_posteriors[c_src]['posteriors']
                W = torch.zeros(self.num_states, self.num_states, device=self.device)  # [K_src, K_dst]
                count = torch.zeros(self.num_states, device=self.device)

                for n in range(len(data)):
                    xi_dst = torch.exp(posteriors[n]['log_xi'])        # [T-1, K, K]
                    gamma_src = torch.exp(src_posteriors[n]['log_gamma'])  # [T, K]
                    gamma_dst = torch.exp(posteriors[n]['log_gamma'])      # [T, K]

                    # Transition counts from xi, weighted by source chain's gamma at t-1
                    # xi_dst.sum(dim=1) gives [T-1, K_dst] (marginal over source state in dst chain)
                    xi_marginal = xi_dst.sum(dim=1)  # [T-1, K_dst]
                    # gamma_src[:-1] is [T-1, K_src]
                    W += gamma_src[:-1].T @ xi_marginal  # [K_src, K_dst]
                    count += gamma_src[:-1].sum(dim=0)    # [K_src]

                eps = 1e-8
                self.coupling_logits[c_src, c_dst] = W / (count.unsqueeze(1) + eps)

    def fit(self, data: list[np.array], max_iteration: int, delta_likelyhood: float):
        logger.info(f"Starting EM for CoupledHMM with {self.num_chains} chains, "
                     f"max {max_iteration} iterations")
        start_time = time.time()
        prev_ll: float | None = None
        cleaned_data = self.get_cleaned_data(data)
        for iteration in range(int(max_iteration)):
            log_posteriors = self._e_step(cleaned_data)
            self._m_step(cleaned_data, log_posteriors)
            new_ll = self.log_likelihood(cleaned_data)
            logger.info(f"Iteration {iteration}: log-likelihood = {new_ll:.2f}")
            if prev_ll is not None and abs(new_ll - prev_ll) < delta_likelyhood:
                logger.info(f"Converged at iteration {iteration}")
                self.iterations_run = iteration + 1
                break
            prev_ll = new_ll
        else:
            self.iterations_run = int(max_iteration)
        self.final_log_likelihood = new_ll
        self.fit_time = time.time() - start_time

    def log_likelihood(self, data: list[torch.Tensor]) -> float:
        total_ll = 0.0
        K = self.num_states
        for seq in data:
            seq = seq if isinstance(seq, torch.Tensor) else torch.tensor(seq, dtype=torch.long, device=self.device)
            T = len(seq)
            chain_gammas = {i: torch.full((T, K), -np.log(K), device=self.device)
                           for i in range(self.num_chains)}
            for _ in range(5):
                for ci in range(self.num_chains):
                    alpha = self._forward_coupled(seq, ci, chain_gammas)
                    beta = self._backward_coupled(seq, ci, chain_gammas)
                    chain_gammas[ci] = self._compute_gamma(alpha, beta)

            for ci in range(self.num_chains):
                alpha = self._forward_coupled(seq, ci, chain_gammas)
                total_ll += torch.logsumexp(alpha[-1], dim=0).item()
        return total_ll

    def predict_missing(self, sequence: list[int], mask_index: int) -> int:
        seq = torch.tensor(sequence, dtype=torch.long, device=self.device)
        T = len(seq)
        K = self.num_states

        chain_gammas = {i: torch.full((T, K), -np.log(K), device=self.device)
                       for i in range(self.num_chains)}
        for _ in range(5):
            for ci in range(self.num_chains):
                alpha = self._forward_coupled(seq, ci, chain_gammas)
                beta = self._backward_coupled(seq, ci, chain_gammas)
                chain_gammas[ci] = self._compute_gamma(alpha, beta)

        log_vocab_scores = torch.zeros(self.vocab_size, device=self.device)
        for ci, hmm in enumerate(self.chain_hmms):
            gamma_at_mask = torch.exp(chain_gammas[ci][mask_index])  # [K]
            log_vocab_scores += (gamma_at_mask.unsqueeze(1) * hmm.log_emission_probs).sum(dim=0)

        return torch.argmax(log_vocab_scores).item()

    def perplexity(self, dataset: list[list[int]]) -> float:
        cleaned = self.get_cleaned_data(dataset)
        total_ll = self.log_likelihood(cleaned)
        total_tokens = sum(len(seq) for seq in cleaned)
        return np.exp(-total_ll / total_tokens)

    def dump(self) -> None:
        logger.info("Dumping model and metadata...")
        random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        model_name = f"m_{random_id}"
        metadata = {
            "model_type": self.__class__.__name__,
            "vocab_size": self.vocab_size,
            "num_states": self.num_states,
            "num_chains": self.num_chains,
            "device": self.device,
            "initial_initializer": self.initial_initializer_name,
            "transition_initializer": self.transition_initializer_name,
            "emission_initializer": self.emission_initializer_name,
            "iterations_run": self.iterations_run,
            "final_log_likelihood": self.final_log_likelihood,
            "fit_time": self.fit_time,
        }
        Path(f'{trained_model_metadata_path}/{model_name}.json').write_text(json.dumps(metadata))
        with open(f"{trained_model_path}/{model_name}.pkl", "wb") as f:
            pickle.dump(self, f)
