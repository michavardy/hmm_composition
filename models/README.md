# compositional hmm architectures

## Terms
- **switching mode** is a latent variable that selects or blends parameter regimes of the HMM.

## switching_hmm_hard
- multiple discrete modes (regimes), each with its own transition and emission parameters
- a single mode is active at each timestep
- the mode evolves as a latent Markov process
- the model learns both regime-specific dynamics and when transitions between regimes occur
- results in piecewise stationary dynamics

## switching_hmm_soft
- multiple modes (regimes), each with its own transition and emission parameters
- all modes contribute at each timestep via a latent mixture (probability distribution over modes)
- the model learns time-varying weights over regimes
- dynamics are smoothly varying and non-stationary through interpolation between regimes