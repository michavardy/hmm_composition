import json
import pickle
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from utils.setup_logger import get_logger

logger = get_logger("PREPROCESS")

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "utils" / "config.toml"


def load_config() -> dict:
  with CONFIG_PATH.open("rb") as config_file:
    return tomllib.load(config_file)["preprocess"]


def resolve_path(path_value: str) -> Path:
  path = Path(path_value)
  if path.is_absolute():
    return path
  return ROOT_DIR / path


def load_corpus(config: dict) -> list[str]:
  input_dir = resolve_path(config["input_dir"])
  file_pattern = config["file_pattern"]

  if not input_dir.exists():
    raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

  text_paths = sorted(input_dir.glob(file_pattern))
  if not text_paths:
    raise FileNotFoundError(
      f"No input files matched pattern '{file_pattern}' in {input_dir}"
    )

  logger.info("Loading %s text files from %s", len(text_paths), input_dir)
  encoding = config["encoding"]
  return [path.read_text(encoding=encoding) for path in text_paths]


def compile_cleaning_pattern(keep_punctuation: str) -> re.Pattern[str]:
  allowed_punctuation = re.escape(keep_punctuation)
  return re.compile(rf"[^a-z0-9\s{allowed_punctuation}]")


def clean_text(text: str, config: dict, cleaning_pattern: re.Pattern[str]) -> str:
  normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
  if config["lowercase"]:
    normalized_text = normalized_text.lower()

  normalized_text = cleaning_pattern.sub(" ", normalized_text)
  normalized_text = re.sub(r"([.,?])", r" \1 ", normalized_text)

  if config["normalize_whitespace"]:
    normalized_text = re.sub(r"[^\S\n]+", " ", normalized_text)
    normalized_text = re.sub(r"\n{3,}", "\n\n", normalized_text)

  return normalized_text.strip()


def split_sequences(cleaned_text: str, config: dict) -> list[str]:
  split_mode = config["split_mode"]
  if split_mode == "sentence":
    parts = re.split(config["sentence_split_pattern"], cleaned_text)
  elif split_mode == "line":
    parts = re.split(config["line_split_pattern"], cleaned_text)
  else:
    raise ValueError(f"Unsupported split_mode: {split_mode}")

  return [part.strip() for part in parts if part.strip()]


def tokenize_sequences(sequences: list[str], token_pattern: re.Pattern[str]) -> list[list[str]]:
  tokenized_sequences: list[list[str]] = []
  for sequence in sequences:
    tokens = token_pattern.findall(sequence)
    if tokens:
      tokenized_sequences.append(tokens)
  return tokenized_sequences


def build_vocabulary(tokenized_sequences: list[list[str]], config: dict) -> tuple[dict[str, int], dict[int, str], Counter[str]]:
  token_counts: Counter[str] = Counter()
  for sequence in tokenized_sequences:
    token_counts.update(sequence)

  special_tokens = list(config["special_tokens"])
  rare_token_cutoff = int(config["rare_token_cutoff"])
  max_vocab_size = int(config["max_vocab_size"])
  reserved_slots = len(special_tokens)

  frequent_tokens = [
    token
    for token, count in token_counts.items()
    if count >= rare_token_cutoff and token not in special_tokens
  ]
  frequent_tokens.sort(key=lambda token: (-token_counts[token], token))
  if max_vocab_size > 0:
    frequent_tokens = frequent_tokens[: max(0, max_vocab_size - reserved_slots)]

  vocabulary_tokens = special_tokens + frequent_tokens
  word_to_id = {token: index for index, token in enumerate(vocabulary_tokens)}
  id_to_word = {index: token for token, index in word_to_id.items()}
  return word_to_id, id_to_word, token_counts


def replace_rare_tokens(
  tokenized_sequences: list[list[str]],
  word_to_id: dict[str, int],
  unk_token: str,
) -> list[list[str]]:
  return [
    [token if token in word_to_id else unk_token for token in sequence]
    for sequence in tokenized_sequences
  ]


def encode_sequences(
  tokenized_sequences: list[list[str]],
  word_to_id: dict[str, int],
  config: dict,
) -> list[np.ndarray]:
  unk_token = config["special_tokens"][1]
  unk_id = word_to_id[unk_token]
  min_sequence_length = int(config["min_sequence_length"])
  numpy_dtype = np.dtype(config["numpy_dtype"])

  encoded_sequences: list[np.ndarray] = []
  for sequence in tokenized_sequences:
    encoded = np.array(
      [word_to_id.get(token, unk_id) for token in sequence],
      dtype=numpy_dtype,
    )
    if len(encoded) >= min_sequence_length:
      encoded_sequences.append(encoded)

  return encoded_sequences


def split_dataset(encoded_sequences: list[np.ndarray], config: dict) -> tuple[list[np.ndarray], list[np.ndarray]]:
  if len(encoded_sequences) < 2:
    raise ValueError("Need at least two sequences to perform a train/test split")

  train_data, test_data = train_test_split(
    encoded_sequences,
    test_size=float(config["test_size"]),
    random_state=int(config["random_state"]),
    shuffle=bool(config["shuffle"]),
  )
  return list(train_data), list(test_data)


def save_pickle(path: Path, payload: object) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("wb") as output_file:
    pickle.dump(payload, output_file, protocol=pickle.HIGHEST_PROTOCOL)


def save_json(path: Path, payload: object) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_artifacts(
  train_data: list[np.ndarray],
  test_data: list[np.ndarray],
  word_to_id: dict[str, int],
  id_to_word: dict[int, str],
  token_counts: Counter[str],
  config: dict,
) -> None:
  output_dir = resolve_path(config["output_dir"])
  vocab_payload = {
    "word_to_id": word_to_id,
    "id_to_word": {str(index): token for index, token in id_to_word.items()},
  }
  metadata_payload = {
    "train_sequences": len(train_data),
    "test_sequences": len(test_data),
    "vocab_size": len(word_to_id),
    "rare_token_cutoff": int(config["rare_token_cutoff"]),
    "max_vocab_size": int(config["max_vocab_size"]),
    "top_tokens": token_counts.most_common(25),
  }

  save_pickle(output_dir / config["train_output"], train_data)
  save_pickle(output_dir / config["test_output"], test_data)
  save_json(output_dir / config["vocab_json_output"], vocab_payload)
  save_pickle(output_dir / config["vocab_pickle_output"], vocab_payload)
  save_pickle(output_dir / config["metadata_output"], metadata_payload)

  logger.info("Saved processed artifacts to %s", output_dir)


def run_preprocess() -> None:
  logger.info("Starting preprocessing pipeline")
  config = load_config()
  raw_documents = load_corpus(config)
  cleaning_pattern = compile_cleaning_pattern(config["keep_punctuation"])
  token_pattern = re.compile(config["token_pattern"])

  segmented_sequences: list[str] = []
  for document in raw_documents:
    cleaned_document = clean_text(document, config, cleaning_pattern)
    segmented_sequences.extend(split_sequences(cleaned_document, config))

  tokenized_sequences = tokenize_sequences(segmented_sequences, token_pattern)
  word_to_id, id_to_word, token_counts = build_vocabulary(tokenized_sequences, config)
  normalized_sequences = replace_rare_tokens(
    tokenized_sequences,
    word_to_id,
    config["special_tokens"][1],
  )
  encoded_sequences = encode_sequences(normalized_sequences, word_to_id, config)
  train_data, test_data = split_dataset(encoded_sequences, config)
  save_artifacts(train_data, test_data, word_to_id, id_to_word, token_counts, config)

  logger.info(
    "Completed preprocessing: %s total sequences, vocab size %s, train=%s, test=%s",
    len(encoded_sequences),
    len(word_to_id),
    len(train_data),
    len(test_data),
  )


if __name__ == "__main__":
  run_preprocess()