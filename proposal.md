# Evaluating Compositional Hidden Markov Models for Next-Token Prediction in the Shakespeare Corpus

---

## Research Objective

To evaluate whether compositional Hidden Markov Models (HMMs)—including Cloned HMMs (CHMM), Factorial HMMs (FHMM), Fractional HMMs (fHMM), and Switching HMMs—can outperform vanilla HMMs in next-token prediction, given a similar state size. We also aim to quantify the effect of the number of states on predictive performance.

---

## Corpus

**Shakespeare Corpus (complete works)**

* **Justification:**

  * Limited vocabulary
  * Well-studied in NLP literature
  * Structured yet manageable text sequences



## Baselines

* **N-grams**

  * 3-gram (Kneser-Ney)
  * 5-gram (Kneser-Ney)

* **Vanilla HMM**

  * states = 50
  * states = 100
  * states = 200
  
* **LSTM reference**

  * 2 × 50 layers

---

## Proposed Models

1. **Cloned HMM (CHMM)**

   * greedy, states - 100
   * Sparse emission matrix,  states - 100

2. **Factorial HMM (FHMM)**

   * chains 2, states - 50
   * chains 3, states - 33
   * chains 4, states - 20

*Note*: attempt to equalize latent combinations

3. **Fractional HMM (fHMM)**

   * Memory window a = 5, states - 100
   * Memory window a = 10, states - 100
   * Memory window a = 20, states - 100


1. **Switching HMM**

   * modes: 2 × 50 states, total - 100
   * modes: 4 x 20 states, total - 100


---

## Research Methodology

### Preprocessing
    - Cleaning
      - lower case
      - remove punctuation (except for ".", ",", "?" )
      - Normalize whitespace and line breaks.
    - Tokenize
      - word level tokenization
      - rare word removal
        - <UNK>
        - words appearing fewer than 5 times -> <UNK>\
    - Vocabulary
      - limit total vocabulary to 5-10K words
      - map all others to <UNK>
    - Sequence Segmentation
      - split by line or sentence for independent sequences
    - encoding
      - map each token to integer ID 
    - Split
      - shuffle at sequence level
      - split 80 / 20 train test



### Training

* HMM-based models: EM (Baum-Welch), max 30 iterations, early stopping ΔlogL < 1e-3
* CHMM: greedy splitting, sparse transitions
* FHMM: approximate EM for factorial joint states
* fHMM: memory window truncation to control compute
* LSTM: cross-entropy, early stopping on validation perplexity

### Testing

Missing Word Prediction
for each sequence in 20% test set
    - randomly select one word to hide
    - predict hidden word by marginalizing over hidden states
    - 


Evaluation

Perplexity: (PPL) quantifies model uncertainty on the hidden word
$$
PPL = exp(-\frac{1}{N}\sum_{i=1}^N log P(\text{hidden word}_i | O_i))
$$

### Evaluation Metrics

* Primary: Perplexity (PPL)
* Secondary: Top-5 accuracy
* Compute time per iteration, 
* memory usage

---

## Expected Outcomes

* Compositional HMMs (CHMM, FHMM, fHMM, Switching HMM) outperform vanilla HMMs for the same total state capacity.
* Increasing total states improves performance, but structural design dominates gains at small N.
* CHMM expected to show the strongest improvement due to variable-order memory.
* FHMM and fHMM expected to provide additional gains via distributed and long-term memory, respectively.

## References

Cloned Hidden Markov Models (CHMM): Dedieu, A., Gothoskar, N., Swingle, S., Lehrach, W., Lázaro-Gredilla, M., & George, D. (2019). Learning higher-order sequential structure with cloned HMMs. This source details the use of sparse emission matrices where multiple "clone" hidden states map to a single observation to capture long-range dependencies

Factorial Hidden Markov Models (FHMM): Nepal, A., & Yates, A. (2013). Factorial Hidden Markov Models for Learning Representations of Natural Language. This work introduces discrete-valued FHMMs and variational EM for language modeling, emphasizing distributed state representations

Fractional Hidden Markov Models (fHMM): Elliott, R. J., & Siu, T. K. (2011). Control of discrete-time HMM partially observed under fractional Gaussian noises. This study characterizes models incorporating long-term memory (the "Joseph effect") using fractional differencing in discrete-time HMMs

Switching/Non-Stationary Hidden Markov Models: Xiao, J., Liu, B., & Wang, X. (2005). Principles of Non-stationary Hidden Markov Model and Its Applications to Sequence Labeling Task. Harbin Institute of Technology. This paper proposes relaxing the "stationary hypothesis" to improve predictive power through time-varying transitions and emissions

Infinite Factorial HMMs: Van Gael, J., Teh, Y. W., & Ghahramani, Z. (2009). The Infinite Factorial Hidden Markov Model. This source provides a nonparametric extension for FHMMs, allowing the number of parallel Markov chains to be learned from the data

### Baselines and Evaluation Context

Comparative NLP Metrics: Dai, Y., Gao, Z., Sattar, Y., Dean, S., & Sun, J. J. (2025). Pre-trained Large Language Models Learn Hidden Markov Models In-context. Provides performance benchmarks comparing Viterbi targets, Baum-Welch, LSTMs, and n-grams on HMM-generated sequences

Stochastic NLP Survey: Almutiri, T., & Nadeem, F. (2022). Markov Models Applications in Natural Language Processing: A Survey. Reviews the foundational differences between Markov chains and HMMs in language generation and POS tagging

### Corpus-Specific Implementation
Shakespeare HMM Implementation: Usajid. (7 years ago). HMM-based-Text-Prediction-Generation. GitHub Repository. Provides a blueprint for tokenization, Baum-Welch training, and poem generation specifically using the Shakespeare corpus with varying hidden state counts

