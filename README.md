# HMM Composition

HMM Composition is a small research-oriented project for training and comparing sequence models on a Shakespeare corpus. The codebase focuses on a practical workflow: preprocess text into token sequences, run one or more configured experiments, save metrics and model artifacts, and inspect results in a lightweight local dashboard.

The project includes several Hidden Markov Model variants, an n-gram baseline, and an LSTM baseline. The emphasis is on running repeatable experiments from configuration rather than building a packaged library.

## What This Project Does

- Preprocesses raw Shakespeare text into train and test sequence datasets.
- Builds a shared vocabulary and replaces rare tokens with `<UNK>`.
- Trains one or more configured models from `utils/config.toml`.
- Saves each experiment as a separate run with config, metrics, logs, and a serialized model.
- Optionally logs run history and metrics with Weights & Biases in offline or online mode.
- Serves a simple dashboard for browsing experiment results.

## Repository Layout

```text
hmm_composition/
├── apps/                # Local dashboard and HTML rendering
├── data/                # Raw and processed data artifacts
├── experiments/         # Per-run outputs (config, metrics, model, logs)
├── models/              # Model implementations and registry
├── pipelines/           # Preprocessing and training entry points
├── scripts/             # Small environment helper scripts
├── utils/               # Shared utilities and config
├── proposal.md          # Research notes
└── README.md
```

## Requirements

- Python 3.11 or newer
- A virtual environment is recommended
- Core Python packages used by the project:
	- `numpy`
	- `scikit-learn`
	- `colorama`
- Optional packages:
	- `torch` for LSTM experiments
	- `wandb` for experiment tracking

If you do not already have a dependency file for this project, install the minimum set manually:

```bash
pip install numpy scikit-learn colorama
pip install torch wandb
```

If you do not plan to run the LSTM or Weights & Biases tracking, the second command is optional.

## Setup

Create and activate a virtual environment from the project root.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Then install the dependencies you need.

## Quick Start

### 1. Preprocess the Corpus

This reads raw text files, cleans them, tokenizes sequences, builds the vocabulary, and writes processed artifacts under `data/processed`.

```bash
python pipelines/preprocess.py
```

### 2. Configure Experiments

Edit `utils/config.toml` to control:

- preprocessing settings
- enabled experiments
- model-specific hyperparameters
- dashboard settings
- Weights & Biases settings

Each `[[training.experiments]]` block defines one experiment. Running the training pipeline executes every experiment with `enabled = true`.

### 3. Run Training

```bash
python pipelines/train.py
```

Training creates a new run directory under `experiments/` for each enabled experiment.

### 4. Open the Dashboard

```bash
python apps/dashboard.py
```

By default, the dashboard reads from the configured experiments directory and serves a local page showing run-level metrics such as perplexity, prediction accuracy, subset sizes, and iteration counts.

## Configuration

The main configuration file is `utils/config.toml`.

Important sections:

- `[preprocess]`: corpus loading, tokenization, vocabulary, splitting, and output paths
- `[training]`: processed artifact paths, experiment root directory, random seed, and W&B settings
- `[[training.experiments]]`: one block per experiment
- `[model_*]`: model-specific hyperparameters
- `[dashboard]`: local result viewer settings

### Experiment Subsets

`train_subset` and `test_subset` limit how many sequences an experiment uses. This is mainly useful for:

- faster smoke tests
- debugging model code
- quick baseline comparisons
- reducing runtime for expensive models such as the LSTM

If a subset limit is `0` or larger than the available dataset, the full split is used.

## Models

The project currently includes implementations for:

- HMM
- CHMM
- FHMM
- finite HMM (`f_hmm`)
- switching HMM
- LSTM
- n-gram

The exact set of experiments you run is controlled by `utils/config.toml`.

## Outputs

Each experiment run gets its own directory inside `experiments/`, for example:

```text
experiments/
└── run_001/
		├── config.json
		├── metrics.json
		├── model.pkl
		└── logs/
```

Typical artifacts:

- `config.json`: merged configuration used for that run
- `metrics.json`: train and test metrics, prediction accuracy, and history summary
- `model.pkl`: serialized trained model
- `logs/`: local logs and optional W&B offline files

Processed data artifacts are written to `data/processed/` and typically include:

- train sequences
- test sequences
- vocabulary JSON and pickle files
- preprocessing metadata

## Weights & Biases

The training pipeline can log experiment configuration, history, and final metrics to Weights & Biases.

Relevant config values:

- `wandb_project`
- `wandb_entity`
- `wandb_mode`

By default, the project is configured for offline tracking. That means runs are still recorded locally, but they are not uploaded unless you switch to online mode and sync them.

## Typical Workflow

1. Adjust preprocessing or experiment settings in `utils/config.toml`.
2. Run preprocessing if the corpus or preprocessing settings changed.
3. Run training for all enabled experiments.
4. Inspect `experiments/` outputs or open the dashboard.
5. Repeat with larger subsets or longer runs once the pipeline looks correct.

## Notes

- This repository is organized as a runnable research project, not a published Python package.
- Training runs are configuration-driven and intended to be easy to repeat.
- LSTM experiments are typically the slowest option, especially on CPU and with larger subset sizes.

## Troubleshooting

### `wandb` is not installed

If W&B is unavailable, training still runs. The pipeline logs a warning and skips experiment tracking.

### `torch` is not installed

Only LSTM experiments require PyTorch. HMM and n-gram experiments can still run without it.

### No runs appear in the dashboard

Make sure you have already executed training and that `dashboard.experiments_dir` points to the same directory where run outputs are being written.
