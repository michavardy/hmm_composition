from __future__ import annotations

import json
import pickle
import random
import string
import time
from pathlib import Path

from utils.constants import trained_model_path, trained_model_metadata_path
from utils.decorators import log_time_and_memory
from utils.setup_logger import get_logger
from typing import Any
import numpy as np
import torch
from models.base_hmm import BaseHMMModel
logger = get_logger("base_hmm")

class factorialHMM:
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
        # core config
        self.vocab_size = vocab_size
        self.num_chains = num_chains
        self.initial_initializer_name = initial_initializer
        self.transition_initializer_name = transition_initializer
        self.emission_initializer_name = emission_initializer
        self.iterations_run = 0
        self.final_log_likelihood = None
        self.fit_time = None
        self.num_states = num_states
        self.device = device
        
        # composition: one BaseHMM per chain
        self.chain_hmms: list[BaseHMMModel] = []
        self._initialize_chain_models(
            initial_initializer, transition_initializer, emission_initializer
        )
        
    def _initialize_chain_models(self, initial_initializer, transition_initializer, emission_initializer):
        # Instantiate one BaseHMMModel per chain
        for _ in range(self.num_chains):
            self.chain_hmms.append(
                BaseHMMModel(
                    vocab_size=self.vocab_size,
                    num_states=self.num_states,
                    device=self.device,
                    initial_initializer=initial_initializer,
                    transition_initializer=transition_initializer,
                    emission_initializer=emission_initializer,
                )
            )
        assert len(self.chain_hmms) == self.num_chains, "Chain HMMs not properly initialized"
    
    def get_cleaned_data(self, data: list[np.array]) -> list[torch.Tensor]:
        cleaned_data = [
            seq.detach().clone().to(dtype=torch.long, device=self.device) if isinstance(seq, torch.Tensor) 
            else torch.tensor(seq, dtype=torch.long, device=self.device)
            for seq in data
        ]
        return cleaned_data
    
    def get_log_joint_emissions(self, sequence: torch.Tensor, target_chain: BaseHMMModel, chain_gammas: dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Compute mean-field effective emissions for target_chain by
        adding the expected log-emission from every other chain,
        weighted by that chain's current variational posterior (gamma).

        Args:
            sequence: [T] observed token indices
            target_chain: the BaseHMMModel whose effective emissions we compute
            chain_gammas: {chain_index: gamma [T, K]} current variational posteriors for all chains

        Returns: [T, num_states] effective log-emissions for target_chain.
        """
        # Start with the target chain's own log-emission: [num_states, T] -> [T, num_states]
        effective_emissions = target_chain.log_emission_probs[:, sequence].T.clone()  # [T, K]

        # Add contribution from each other chain using their current variational gamma
        for chain_idx, chain in enumerate(self.chain_hmms):
            if chain is target_chain:
                continue
            gamma = torch.exp(chain_gammas[chain_idx])  # [T, K] probabilities

            # This chain's log-emission for the observed tokens: [num_states, T] -> [T, num_states]
            chain_log_em = chain.log_emission_probs[:, sequence].T  # [T, K]

            # Expected log-emission: sum_j gamma[t,j] * log B_c(o_t | j)  -> [T]
            expected_log_em = (gamma * chain_log_em).sum(dim=1, keepdim=True)  # [T, 1]
            effective_emissions = effective_emissions + expected_log_em  # [T, K]

        return effective_emissions  # [T, num_states]
    
    def _forward(self, sequence: torch.Tensor, joint_emissions: torch.Tensor, hmm: BaseHMMModel) -> torch.Tensor:
        # sequence: [T]  joint_emissions: [T, K]
        alpha = torch.zeros((len(sequence), hmm.num_states), device=self.device)  # [T, K]
        alpha[0] = hmm.log_initial_probs + joint_emissions[0, :]  # [K] + [K] -> [K]
        for t in range(1, len(sequence)):
            prev_alpha = alpha[t-1]  # [K]
            scores = prev_alpha.unsqueeze(1) + hmm.log_transition_probs  # [K,1] + [K,K] -> [K,K]
            alpha[t] = torch.logsumexp(scores, dim=0) + joint_emissions[t, :]  # [K] + [K] -> [K]
        return alpha  # [T, K]
    
    def _backward(self, sequence: torch.Tensor, joint_emissions: torch.Tensor, hmm: BaseHMMModel) -> torch.Tensor:
        # sequence: [T]  joint_emissions: [T, K]
        beta = torch.zeros((len(sequence), hmm.num_states), device=self.device)  # [T, K]
        for t in reversed(range(len(sequence) - 1)):
            next_beta = beta[t+1]  # [K]
            scores = hmm.log_transition_probs + joint_emissions[t+1, :] + next_beta  # [K,K] + [K] + [K] -> [K,K]
            beta[t] = torch.logsumexp(scores, dim=1)  # [K]
        return beta  # [T, K]
    
    def _compute_gamma(self, log_alpha: torch.Tensor, log_beta: torch.Tensor) -> torch.Tensor:
        # log_alpha: [T, K]  log_beta: [T, K]
        log_gamma = log_alpha + log_beta  # [T, K]
        log_gamma -= torch.logsumexp(log_gamma, dim=1, keepdim=True)  # normalize: [T, K] - [T, 1] -> [T, K]
        return log_gamma  # [T, K]
    
    def _compute_xi(self, sequence: torch.Tensor, log_alpha: torch.Tensor, log_beta: torch.Tensor, joint_emissions: torch.Tensor, hmm:BaseHMMModel) -> torch.Tensor:
        # sequence: [T]  log_alpha: [T, K]  log_beta: [T, K]  joint_emissions: [T, K]
        log_xi = torch.zeros(
            (len(sequence) - 1, hmm.num_states, hmm.num_states),
            device=self.device
        )  # [T-1, K, K]
        
        for t in range(len(sequence) - 1):
            scores = (
                log_alpha[t].unsqueeze(1)      # [K, 1]
                + hmm.log_transition_probs     # [K, K]
                + joint_emissions[t+1, :].unsqueeze(0)  # [1, K]
                + log_beta[t+1]                # [K] broadcast -> [K, K]
            )  # [K, K]
            
            log_norm = torch.logsumexp(scores.flatten(), dim=0)  # scalar
            log_xi[t] = scores - log_norm  # [K, K]
    
        return log_xi  # [T-1, K, K]
    
    @torch.no_grad()
    def _e_step(self, data: list[torch.Tensor], num_mf_iterations: int = 5, mf_tol: float = 1e-4) -> list[dict[str, torch.Tensor]]:
        """
        E-step with mean-field coordinate ascent.
        Caches log-probs so repeated @property log_softmax calls are ~free.
        """
        K = self.num_states

        # Cache: set logits = log_probs so subsequent log_softmax(log_probs) ≈ log_probs
        saved_logits = []
        for hmm in self.chain_hmms:
            saved_logits.append((hmm.initial_logits.clone(), hmm.transition_logits.clone(), hmm.emission_logits.clone()))
            hmm.initial_logits = hmm.log_initial_probs
            hmm.transition_logits = hmm.log_transition_probs
            hmm.emission_logits = hmm.log_emission_probs

        chain_posteriors = [
            {'chain_idx': i, 'hmm': self.chain_hmms[i], 'posteriors': []}
            for i in range(self.num_chains)
        ]

        for sequence in data:  # sequence: [T]
            T = len(sequence)
            chain_gammas = {i: torch.full((T, K), -np.log(K), device=self.device) for i in range(self.num_chains)}
            seq_data = {}

            for mf_iter in range(num_mf_iterations):
                max_delta = 0.0
                for ci, hmm in enumerate(self.chain_hmms):
                    eff = self.get_log_joint_emissions(sequence, hmm, chain_gammas)
                    alpha = self._forward(sequence, eff, hmm)
                    beta = self._backward(sequence, eff, hmm)
                    gamma = self._compute_gamma(alpha, beta)

                    delta = (torch.exp(gamma) - torch.exp(chain_gammas[ci])).abs().sum().item()
                    max_delta = max(max_delta, delta)
                    chain_gammas[ci] = gamma
                    seq_data[ci] = (alpha, beta, eff)

                if max_delta < mf_tol:
                    logger.debug(f"MF converged at iter {mf_iter} (delta={max_delta:.6f})")
                    break

            # Compute xi once per chain using converged posteriors
            for ci, hmm in enumerate(self.chain_hmms):
                a, b, e = seq_data[ci]
                chain_posteriors[ci]['posteriors'].append({
                    'log_alpha': a,
                    'log_beta': b,
                    'log_gamma': chain_gammas[ci],
                    'log_xi': self._compute_xi(sequence, a, b, e, hmm),
                })

        # Restore original logits
        for hmm, (init, trans, emit) in zip(self.chain_hmms, saved_logits):
            hmm.initial_logits = init
            hmm.transition_logits = trans
            hmm.emission_logits = emit

        return chain_posteriors
    
    def _m_step(self, data: list[torch.Tensor], chain_posteriors: list[dict]) -> None:
        for chain_post in chain_posteriors:
            hmm = chain_post['hmm']
            hmm._m_step(data, chain_post['posteriors'])
            
    def fit(self, data: list[np.array], max_iteration: int, delta_likelyhood: float):
        logger.info(f"Starting EM training for max {max_iteration} iterations with delta {delta_likelyhood}")
        start_time = time.time()
        prev_log_likelihood: float | None = None
        cleaned_data = self.get_cleaned_data(data)
        for iteration in range(int(max_iteration)):
            #            chains               sequence
            #log_posteriors[0].get('posteriors')[0].keys()  -> dict_keys(['log_alpha', 'log_beta', 'log_gamma', 'log_xi'])
            log_posteriors = self._e_step(cleaned_data)
            self._m_step(cleaned_data, log_posteriors)
        self.iterations_run = int(max_iteration)
        self.final_log_likelihood = self.log_likelihood(cleaned_data)
        self.fit_time = time.time() - start_time
    
    def log_likelihood(self, data: list[torch.Tensor]) -> float:
        """
        Approximate log-likelihood using the mean-field variational bound.
        For each sequence, run MF E-step to get converged gammas, then
        compute the expected complete-data log-likelihood.
        """
        total_ll = 0.0
        K = self.num_states

        for seq in data:
            seq = seq if isinstance(seq, torch.Tensor) else torch.tensor(seq, dtype=torch.long, device=self.device)
            T = len(seq)

            # Run MF to convergence for this sequence
            chain_gammas = {i: torch.full((T, K), -np.log(K), device=self.device) for i in range(self.num_chains)}
            for _ in range(5):
                for ci, hmm in enumerate(self.chain_hmms):
                    eff = self.get_log_joint_emissions(seq, hmm, chain_gammas)
                    alpha = self._forward(seq, eff, hmm)
                    beta = self._backward(seq, eff, hmm)
                    chain_gammas[ci] = self._compute_gamma(alpha, beta)

            # Sum each chain's marginal log-likelihood from its converged alpha
            for ci, hmm in enumerate(self.chain_hmms):
                eff = self.get_log_joint_emissions(seq, hmm, chain_gammas)
                alpha = self._forward(seq, eff, hmm)
                total_ll += torch.logsumexp(alpha[-1], dim=0).item()

        return total_ll

    def predict_missing(self, sequence: list[int], mask_index: int) -> int:
        """
        Predict the missing token at mask_index by summing each chain's
        log-predictive score for every vocab token.
        """
        seq = torch.tensor(sequence, dtype=torch.long, device=self.device)
        T = len(seq)
        K = self.num_states

        # Run MF to get converged gammas (using all observed tokens including placeholder at mask)
        chain_gammas = {i: torch.full((T, K), -np.log(K), device=self.device) for i in range(self.num_chains)}
        for _ in range(5):
            for ci, hmm in enumerate(self.chain_hmms):
                eff = self.get_log_joint_emissions(seq, hmm, chain_gammas)
                alpha = self._forward(seq, eff, hmm)
                beta = self._backward(seq, eff, hmm)
                chain_gammas[ci] = self._compute_gamma(alpha, beta)

        # For each chain, compute state posterior at mask_index, then score each vocab token
        log_vocab_scores = torch.zeros(self.vocab_size, device=self.device)
        for ci, hmm in enumerate(self.chain_hmms):
            gamma_at_mask = torch.exp(chain_gammas[ci][mask_index])  # [K]
            log_vocab_scores += (gamma_at_mask.unsqueeze(1) * hmm.log_emission_probs).sum(dim=0)  # [V]

        return torch.argmax(log_vocab_scores).item()

    def perplexity(self, dataset: list[list[int]]) -> float:
        total_tokens = 0
        total_ll = 0.0
        cleaned = self.get_cleaned_data(dataset)
        total_ll = self.log_likelihood(cleaned)
        total_tokens = sum(len(seq) for seq in cleaned)
        avg_ll = total_ll / total_tokens
        return np.exp(-avg_ll)

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
