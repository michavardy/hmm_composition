from sympy import sequence
import torch
import torch.nn.functional as F
from models.base_initializer import (
    ZerosInitializer,
    RandomInitializer,
    SmallRandomInitializer,
    InitializerType,
)
import numpy as np
#from pipelines.train import load_config, load_processed_artifacts
import random
from utils.decorators import log_time_and_memory
from utils.setup_logger import get_logger


logger = get_logger("base_hmm")


class BaseHMMModel:
    def __init__(
        self,
        vocab_size: int,
        num_states: int,
        initial_initializer: InitializerType = ZerosInitializer(),
        transition_initializer: InitializerType = ZerosInitializer(),
        emission_initializer: InitializerType = ZerosInitializer(),
        device="cpu",
    ):
        self.vocab_size = int(vocab_size)
        self.num_states = int(num_states)
        self.device = device
        self.initialize(
            initial_initializer, transition_initializer, emission_initializer
        )

    def initialize(
        self, initial_initializer, transition_initializer, emission_initializer
    ):
        self.initial_logits = initial_initializer(
            (self.num_states,), device=self.device
        )
        self.transition_logits = transition_initializer(
            (self.num_states, self.num_states), device=self.device
        )
        self.emission_logits = emission_initializer(
            (self.num_states, self.vocab_size), device=self.device
        )

    @property
    def log_initial_probs(self):
        return F.log_softmax(self.initial_logits, dim=0)  # (K,)

    @property
    def log_transition_probs(self):
        return F.log_softmax(self.transition_logits, dim=1)  # (K, K)

    @property
    def log_emission_probs(self):
        return F.log_softmax(self.emission_logits, dim=1)  # (K, V)
    
    def _forward(self, sequence: torch.Tensor) -> torch.Tensor:
        alpha = torch.zeros((len(sequence), self.num_states), device=self.device)
        alpha[0] = self.log_initial_probs + self.log_emission_probs[:, sequence[0]]
        for t in range(1, len(sequence)):
            prev_alpha = alpha[t-1]
            scores = prev_alpha.unsqueeze(1) + self.log_transition_probs
            alpha[t] = torch.logsumexp(scores, dim=0) + self.log_emission_probs[:, sequence[t]]
        return alpha
    
    def _backward(self, sequence: torch.Tensor) -> torch.Tensor:
        beta = torch.zeros((len(sequence), self.num_states), device=self.device)
        for t in reversed(range(len(sequence) - 1)):
            next_beta = beta[t+1]
            scores = self.log_transition_probs + self.log_emission_probs[:, sequence[t+1]] + next_beta
            beta[t] = torch.logsumexp(scores, dim=1)
        return beta
            
    def _compute_gamma(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        log_gamma = alpha + beta
        log_gamma -= torch.logsumexp(log_gamma, dim=1, keepdim=True)
        return log_gamma
    
    def _compute_xi(self, sequence, alpha, beta):
        log_xi = torch.zeros(
            (len(sequence) - 1, self.num_states, self.num_states),
            device=self.device
        )
        
        for t in range(len(sequence) - 1):
            scores = (
                alpha[t].unsqueeze(1)
                + self.log_transition_probs
                + self.log_emission_probs[:, sequence[t+1]]
                + beta[t+1]
            )
            
            log_norm = torch.logsumexp(scores.flatten(), dim=0)
            log_xi[t] = scores - log_norm  # stay in log-space
    
        return log_xi
    
    def _sanity_check(self, log_posteriors, atol=1e-2):
        for idx, post in enumerate(log_posteriors):
            gamma = torch.exp(post['log_gamma'])  # (T, K)
            xi = torch.exp(post['log_xi'])       # (T-1, K, K)
            # Check gamma sums to 1 per timestep
            assert torch.allclose(gamma.sum(dim=1), torch.ones_like(gamma[:, 0]), atol=atol), f"[Gamma FAIL] sequence {idx}"
            # Check xi sums to 1 per timestep
            assert torch.allclose(xi.sum(dim=(1, 2)), torch.ones_like(xi[:, 0, 0]), atol=atol), f"[Xi FAIL] sequence {idx}"
            # Optional: check xi row sums match gamma[:-1]
            assert torch.allclose(xi.sum(dim=2), gamma[:-1], atol=atol), f"[Xi-Gamma consistency FAIL] sequence {idx}"
    
    def _calculate_initial_prob_distribution(self, log_posteriors: list[dict[str,torch.Tensor]]) -> None:
        gamma_0 = torch.stack([post['log_gamma'][0] for post in log_posteriors], dim=0)
        log_initial_probs = torch.logsumexp(gamma_0, dim=0) - np.log(len(log_posteriors))
        self.initial_logits.data = log_initial_probs  # you can assign logits directly
        # sanity check
        init_sum = torch.exp(self.initial_logits).sum()
        assert torch.allclose(init_sum, torch.tensor(1.0, device=self.device), atol=1e-4), f"Initial probs do not sum to 1: {init_sum}"
        
    def _calculate_transition_prob_distribution(self, log_posteriors: list[dict[str,torch.Tensor]]) -> None:
        sum_xi = torch.zeros((self.num_states, self.num_states), device=self.device)
        sum_gamma = torch.zeros((self.num_states,), device=self.device)
        for post in log_posteriors:
            xi = torch.exp(post['log_xi'])  # (T-1, K, K)
            gamma = torch.exp(post['log_gamma'])  # (T, K)
            sum_xi += xi.sum(dim=0)             # sum over time
            sum_gamma += gamma[:-1].sum(dim=0)  # sum over time excluding last
        # convert to log-space for logits
        self.transition_logits.data = torch.log(sum_xi / sum_gamma.unsqueeze(1))
        # sanity check
        trans_sum = torch.exp(self.transition_logits).sum(dim=1)
        assert torch.allclose(trans_sum, torch.ones_like(trans_sum), atol=1e-4), f"Transition rows do not sum to 1: {trans_sum}"
        
    def _calculate_emission_prob_distribution(self, data: list[torch.Tensor], log_posteriors: list[dict[str,torch.Tensor]]) -> None:
        sum_gamma_obs = torch.zeros((self.num_states, self.vocab_size), device=self.device)
        sum_gamma_total = torch.zeros((self.num_states,), device=self.device)
        
        for seq, post in zip(data, log_posteriors):
            gamma = torch.exp(post['log_gamma'])  # (T, K)
            for t, obs in enumerate(seq):
                sum_gamma_obs[:, obs] += gamma[t]
            sum_gamma_total += gamma.sum(dim=0)
        
        self.emission_logits.data = torch.log(sum_gamma_obs / sum_gamma_total.unsqueeze(1))
        # sanity check
        emiss_sum = torch.exp(self.emission_logits).sum(dim=1)
        assert torch.allclose(emiss_sum, torch.ones_like(emiss_sum), atol=1e-4), f"Emission rows do not sum to 1: {emiss_sum}"

    def _m_step(self, data: list[torch.Tensor], log_posteriors: list[dict[str,torch.Tensor]]) -> None:
        self._calculate_initial_prob_distribution(log_posteriors)
        self._calculate_transition_prob_distribution(log_posteriors)
        self._calculate_emission_prob_distribution(data, log_posteriors)

    @log_time_and_memory(logger)
    def _e_step(self, data: list[torch.Tensor]) -> list[dict[str,torch.Tensor]]:
        log_posteriors = []
        for seq in data:
            seq_post = {}
            seq_post['log_alpha'] = self._forward(seq)
            seq_post['log_beta'] = self._backward(seq)
            seq_post['log_gamma'] = self._compute_gamma(seq_post['log_alpha'], seq_post['log_beta'])
            seq_post['log_xi'] = self._compute_xi(seq, seq_post['log_alpha'], seq_post['log_beta'])
            log_posteriors.append(seq_post)
        self._sanity_check(log_posteriors)
        return log_posteriors
    
    def _exit_condition(self, iteration: int, prev_log_likelihood: float | None, new_log_likelihood: float, delta_likelyhood: float) -> bool:
        if prev_log_likelihood is None:
            return False
        exit =  abs(new_log_likelihood - prev_log_likelihood) < delta_likelyhood
        if exit:
            logger.info(f"Converged at iteration {iteration} with delta {abs(new_log_likelihood - prev_log_likelihood)}")
        return exit
    
    def fit(self, data: list[np.array], max_iteration: int, delta_likelyhood: float):
        logger.info(f"Starting EM training for max {max_iteration} iterations with delta {delta_likelyhood}")
        prev_log_likelihood: float | None = None
        cleaned_data = [
            seq.detach().clone().to(dtype=torch.long, device=self.device) if isinstance(seq, torch.Tensor) 
            else torch.tensor(seq, dtype=torch.long, device=self.device)
            for seq in data
        ]
        for iteration in range(int(max_iteration)):
            log_posteriors = self._e_step(cleaned_data)
            self._m_step(cleaned_data, log_posteriors)
            new_log_likelihood = self.log_likelihood(cleaned_data)
            if self._exit_condition(iteration, prev_log_likelihood, new_log_likelihood, delta_likelyhood):
                break
            prev_log_likelihood = new_log_likelihood
    
    def emission_logprob_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Returns log P(sequence | HMM) for each time step (sum over states)
        """
        alpha = self._forward(sequence)  # (T, K)
        log_probs = torch.logsumexp(alpha, dim=1)  # (T,)
        return log_probs
    
    def log_likelihood(self, data: list[torch.Tensor]) -> float:
        """
        Computes total log-likelihood of the dataset under current parameters.
        """
        total_log_lik = 0.0
        for seq in data:
            alpha = self._forward(seq)          # (T, K)
            seq_log_lik = torch.logsumexp(alpha[-1], dim=0)  # sum over final states
            total_log_lik += seq_log_lik.item()  # convert to float
        return total_log_lik
    
    def predict_missing(self, sequence: list[int], mask_index: int) -> int:
        """
        Predict the missing symbol at mask_index in sequence.
        sequence: list of integers (word indices)
        mask_index: int, position of missing observation
        Returns: predicted index
        """
        seq = torch.tensor(sequence, dtype=torch.long, device=self.device)
        T = len(seq)
        
        # Forward pass up to mask_index
        alpha = torch.zeros((T, self.num_states), device=self.device)
        alpha[0] = self.log_initial_probs + self.log_emission_probs[:, seq[0]]
        for t in range(1, T):
            if t == mask_index:
                # Skip using the true observation at mask_index
                # Use logsumexp over previous + transition only
                prev_alpha = alpha[t-1]
                scores = prev_alpha.unsqueeze(1) + self.log_transition_probs
                alpha[t] = torch.logsumexp(scores, dim=0)
            else:
                prev_alpha = alpha[t-1]
                scores = prev_alpha.unsqueeze(1) + self.log_transition_probs
                alpha[t] = torch.logsumexp(scores, dim=0) + self.log_emission_probs[:, seq[t]]
        
        # Backward pass from mask_index
        beta = torch.zeros((T, self.num_states), device=self.device)
        for t in reversed(range(T-1)):
            if t == mask_index:
                next_beta = beta[t+1]
                scores = self.log_transition_probs + next_beta
                beta[t] = torch.logsumexp(scores, dim=1)
            else:
                next_beta = beta[t+1]
                scores = self.log_transition_probs + self.log_emission_probs[:, seq[t+1]] + next_beta
                beta[t] = torch.logsumexp(scores, dim=1)
        
        # Compute posterior for missing index
        log_gamma = alpha[mask_index] + beta[mask_index]  # (num_states,)
        
        # Now compute likelihood of each possible symbol
        vocab_post = []
        for v in range(self.vocab_size):
            log_emit = self.log_emission_probs[:, v]
            vocab_post.append(torch.logsumexp(alpha[mask_index-1].unsqueeze(1) + self.log_transition_probs + log_emit + beta[mask_index], dim=0))
        
        vocab_post = torch.stack(vocab_post)  # (V,)
        predicted_index = torch.argmax(vocab_post).item()
        return predicted_index
    
    def perplexity(self, dataset: list[list[int]]) -> float:
        """
        Computes perplexity of the dataset under current HMM parameters.
        dataset: list of sequences (list of indices)
        """
        total_tokens = 0
        total_log_lik = 0.0
        
        for seq in dataset:
            seq_tensor = torch.tensor(seq, dtype=torch.long, device=self.device)
            total_tokens += len(seq)
            alpha = self._forward(seq_tensor)
            seq_log_lik = torch.logsumexp(alpha[-1], dim=0).item()
            total_log_lik += seq_log_lik
        
        avg_log_lik = total_log_lik / total_tokens
        ppl = np.exp(-avg_log_lik)
        return ppl

if __name__ == "__main__":
    pass
    
    #config = load_config()
    #train_data, test_data, vocab = load_processed_artifacts(config["training"])
    #hmm = BaseHMMModel(vocab_size=len(vocab["word_to_id"]), num_states=50)
    #train_data = random.sample(train_data, 6000)
    #hmm.fit(train_data, max_iteration=100, delta_likelyhood=1e-3)
    #sentence = ["to", "be", "or", "not", "to"]
    #index_list = [vocab["word_to_id"].get(word, vocab["word_to_id"]["<UNK>"]) for word in sentence]
    #missing_index = hmm.predict_missing(index_list, mask_index=4)  
    #missing_word = vocab["id_to_word"].get(str(missing_index), "<UNK>") 
    #breakpoint()
