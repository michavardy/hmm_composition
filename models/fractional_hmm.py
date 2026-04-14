import torch
import torch.nn.functional as F
import numpy as np
from models.base_hmm import BaseHMMModel
from utils.setup_logger import get_logger

logger = get_logger("fractional_hmm")


class FractionalHMM(BaseHMMModel):
    """
    Fractional-order HMM using Grünwald-Letnikov weights for power-law memory decay.
    
    At each timestep, alpha[t] depends on a weighted combination of
    alpha[t-1], ..., alpha[t-a] where the weights follow the GL coefficients
    parameterized by fractional order d in (0, 1).
      d=0 → standard HMM (only alpha[t-1] matters)
      d→1 → longer memory (past states decay more slowly)
    """
    def __init__(self, num_states, vocab_size, memory_window=5, d=0.3, device="cpu"):
        super().__init__(vocab_size=vocab_size, num_states=num_states, device=device)
        self.memory_window = memory_window
        self.d = d
        self._gl_weights = self._compute_gl_weights()  # [a] in log-space

    def _compute_gl_weights(self) -> torch.Tensor:
        """
        Compute Grünwald-Letnikov weights w_tau for tau = 1..memory_window.
        w_1 = 1,  w_tau = w_{tau-1} * (tau - 1 - d) / tau
        Returns log-normalized weights: [memory_window]
        """
        w = torch.zeros(self.memory_window)
        w[0] = 1.0
        for tau in range(1, self.memory_window):
            w[tau] = w[tau - 1] * (tau - self.d) / (tau + 1)
        # Normalize and convert to log-space
        w = w / w.sum()
        return torch.log(w).to(self.device)

    def _forward(self, sequence: torch.Tensor) -> torch.Tensor:
        T = len(sequence)
        K = self.num_states
        alpha = torch.zeros((T, K), device=self.device)  # [T, K]
        alpha[0] = self.log_initial_probs + self.log_emission_probs[:, sequence[0]]  # [K]

        for t in range(1, T):
            window = min(t, self.memory_window)
            # Stack past alphas: [window, K]
            past_alphas = alpha[t - window:t].flip(0)  # most recent first, matching w_1..w_window
            log_w = self._gl_weights[:window]  # [window]

            # Weighted combination: log( sum_tau w_tau * sum_i alpha[t-tau,i] * A[i,j] )
            # = logsumexp over (tau, i) of { log_w[tau] + alpha[t-tau, i] + A[i,j] }
            scores = (
                log_w[:, None, None]           # [window, 1, 1]
                + past_alphas[:, :, None]      # [window, K, 1]
                + self.log_transition_probs    # [K, K] broadcast
            )  # [window, K, K]
            alpha[t] = torch.logsumexp(scores.reshape(-1, K), dim=0) + self.log_emission_probs[:, sequence[t]]

        return alpha  # [T, K]

    def _backward(self, sequence: torch.Tensor) -> torch.Tensor:
        T = len(sequence)
        K = self.num_states
        beta = torch.zeros((T, K), device=self.device)  # [T, K]

        for t in range(T - 2, -1, -1):
            window = min(T - 1 - t, self.memory_window)
            # Future betas: [window, K]
            future_betas = beta[t + 1:t + 1 + window]  # nearest future first
            log_w = self._gl_weights[:window]  # [window]

            # log( sum_tau w_tau * sum_j A[i,j] * B[j, o_{t+tau}] * beta[t+tau, j] )
            future_obs = sequence[t + 1:t + 1 + window]  # [window]
            future_emit = self.log_emission_probs[:, future_obs].T  # [window, K]

            scores = (
                log_w[:, None, None]           # [window, 1, 1]
                + self.log_transition_probs    # [K, K] broadcast
                + future_emit[:, None, :]      # [window, 1, K]
                + future_betas[:, None, :]     # [window, 1, K]
            )  # [window, K, K]
            # Marginalize over (tau, j) for each source state i
            beta[t] = torch.logsumexp(scores.permute(0, 2, 1).reshape(-1, K), dim=0)

        return beta  # [T, K]

    # _compute_gamma inherited from BaseHMMModel — works as-is since alpha+beta are consistent

    def _compute_xi(self, sequence, alpha, beta):
        """
        Xi for the fractional model.
        Since transitions are shared across all lags and weighted by GL coefficients,
        xi[t] is the posterior over (i,j) for the t→t+1 transition weighted by w_1
        (the dominant lag). This keeps M-step updates for A[i,j] well-defined.
        """
        T = len(sequence)
        K = self.num_states
        log_xi = torch.zeros((T - 1, K, K), device=self.device)  # [T-1, K, K]

        for t in range(T - 1):
            scores = (
                alpha[t].unsqueeze(1)                         # [K, 1]
                + self.log_transition_probs                   # [K, K]
                + self.log_emission_probs[:, sequence[t + 1]] # [K]
                + beta[t + 1]                                 # [K]
            )  # [K, K]
            log_norm = torch.logsumexp(scores.flatten(), dim=0)
            log_xi[t] = scores - log_norm

        return log_xi  # [T-1, K, K]

    # _e_step, _m_step, _sanity_check, fit, _exit_condition all inherited from BaseHMMModel

    def log_likelihood(self, data: list[torch.Tensor]) -> float:
        total_ll = 0.0
        for seq in data:
            alpha = self._forward(seq)  # [T, K]
            total_ll += torch.logsumexp(alpha[-1], dim=0).item()
        return total_ll

    def predict_missing(self, sequence: list[int], mask_index: int) -> int:
        seq = torch.tensor(sequence, dtype=torch.long, device=self.device)
        T = len(seq)
        K = self.num_states

        # Forward pass — skip emission at mask_index
        alpha = torch.zeros((T, K), device=self.device)
        alpha[0] = self.log_initial_probs + (self.log_emission_probs[:, seq[0]] if mask_index != 0 else 0.0)

        for t in range(1, T):
            window = min(t, self.memory_window)
            past_alphas = alpha[t - window:t].flip(0)
            log_w = self._gl_weights[:window]
            scores = (
                log_w[:, None, None]
                + past_alphas[:, :, None]
                + self.log_transition_probs
            )
            combined = torch.logsumexp(scores.reshape(-1, K), dim=0)
            if t != mask_index:
                alpha[t] = combined + self.log_emission_probs[:, seq[t]]
            else:
                alpha[t] = combined  # no emission at masked position

        # Backward pass — skip emission at mask_index
        beta = torch.zeros((T, K), device=self.device)
        for t in range(T - 2, -1, -1):
            window = min(T - 1 - t, self.memory_window)
            future_obs = seq[t + 1:t + 1 + window]
            future_emit = self.log_emission_probs[:, future_obs].T  # [window, K]
            future_betas = beta[t + 1:t + 1 + window]
            log_w = self._gl_weights[:window]

            # Zero out emission for masked future positions
            mask_offsets = torch.arange(t + 1, t + 1 + window, device=self.device)
            emit_contrib = future_emit.clone()
            emit_contrib[mask_offsets == mask_index] = 0.0

            scores = (
                log_w[:, None, None]
                + self.log_transition_probs
                + emit_contrib[:, None, :]
                + future_betas[:, None, :]
            )
            beta[t] = torch.logsumexp(scores.reshape(-1, K), dim=0)

        # Score each vocab token at mask_index
        log_state_post = alpha[mask_index] + beta[mask_index]  # [K]
        log_vocab_scores = log_state_post.unsqueeze(1) + self.log_emission_probs  # [K, V]
        log_vocab_scores = torch.logsumexp(log_vocab_scores, dim=0)  # [V]
        return torch.argmax(log_vocab_scores).item()

    def _get_metadata(self) -> dict:
        metadata = super()._get_metadata()
        metadata["memory_window"] = self.memory_window
        metadata["d"] = self.d
        return metadata

    def perplexity(self, dataset: list[list[int]]) -> float:
        total_tokens = 0
        total_ll = 0.0
        for seq in dataset:
            seq_t = torch.tensor(seq, dtype=torch.long, device=self.device)
            total_tokens += len(seq)
            alpha = self._forward(seq_t)
            total_ll += torch.logsumexp(alpha[-1], dim=0).item()
        return np.exp(-total_ll / total_tokens)