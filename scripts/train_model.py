from pipelines.train import load_config, load_processed_artifacts
from models.hmm import HMMModel
from models.switching_hmm import SwitchingHMM
from models.fractional_hmm import FractionalHMM
from models.clone_hmm import CloneHMM
from models.factorial_hmm import factorialHMM
from models.hierarchical_hmm import HierarchicalHMM
from models.mixture_hmm import MixtureHMM
from models.coupled_hmm import CoupledHMM
import random
import argparse
from utils.setup_logger import get_logger
logger = get_logger(__name__)

ModelType = HMMModel | SwitchingHMM | FractionalHMM | CloneHMM | factorialHMM | HierarchicalHMM | MixtureHMM | CoupledHMM

"""
python scripts/train_model.py --model factorial_hmm
"""

def get_cli_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train HMM model")
    parser.add_argument("--model",  default="hmm", help="Model type to train (default: hmm)")
    return parser.parse_args()

def get_clone_hmm_model(args: argparse.Namespace, vocab: dict) -> ModelType:
    return  CloneHMM(
        vocab_size=len(vocab["word_to_id"]),
        num_states=10,
        num_clones=2,
    )

def get_hmm_model(args: argparse.Namespace, vocab: dict) -> ModelType:
    return  HMMModel(
        vocab_size=len(vocab["word_to_id"]),
        num_states=50,
    )

def get_switching_hmm_model(args: argparse.Namespace, vocab: dict) -> ModelType:
    return  SwitchingHMM(
        vocab_size=len(vocab["word_to_id"]),
        num_states=10,
        num_modes=2,
    )

def get_fractional_hmm_model(args: argparse.Namespace, vocab: dict) -> ModelType:
    return  FractionalHMM(
        vocab_size=len(vocab["word_to_id"]),
        num_states=10,
        memory_window=5,
    )

def get_factorial_hmm_model(args: argparse.Namespace, vocab: dict) -> ModelType:
    return  factorialHMM(
        vocab_size=len(vocab["word_to_id"]),
        num_states=10,
        num_chains=2,
    )

def get_hierarchical_hmm_model(args: argparse.Namespace, vocab: dict) -> ModelType:
    return  HierarchicalHMM(
        vocab_size=len(vocab["word_to_id"]),
        num_states=5,
        num_productions=3,
    )

def get_mixture_hmm_model(args: argparse.Namespace, vocab: dict) -> ModelType:
    return  MixtureHMM(
        vocab_size=len(vocab["word_to_id"]),
        num_states=10,
        num_components=3,
    )

def get_coupled_hmm_model(args: argparse.Namespace, vocab: dict) -> ModelType:
    return  CoupledHMM(
        vocab_size=len(vocab["word_to_id"]),
        num_states=10,
        num_chains=2,
    )

model_map = {
    "hmm": get_hmm_model,
    "switching_hmm": get_switching_hmm_model,
    "fractional_hmm": get_fractional_hmm_model,
    "clone_hmm": get_clone_hmm_model,
    "factorial_hmm": get_factorial_hmm_model,
    "hierarchical_hmm": get_hierarchical_hmm_model,
    "mixture_hmm": get_mixture_hmm_model,
    "coupled_hmm": get_coupled_hmm_model,
}

def main(args: argparse.Namespace) -> tuple[ModelType, dict]:
    config = load_config()
    train_data, test_data, vocab = load_processed_artifacts(config["training"])
    train_data = random.sample(train_data, 6000)
    model = model_map[args.model](args, vocab)  
    model.fit(train_data, max_iteration=2, delta_likelyhood=1e-3)
    return model, vocab

def test_model(model, vocab):
    #sentence = ["to", "be", "or", "not", "to"]
    s2 = [ "is", "the", "soul", "of", "wit"]
    index_list = [vocab["word_to_id"].get(word, vocab["word_to_id"]["<UNK>"]) for word in s2]
    missing_index = model.predict_missing(index_list, mask_index=0)  
    missing_word = vocab["id_to_word"].get(str(missing_index), "<UNK>") 
    logger.info(f"Predicted missing word {missing_word} in {s2}")

if __name__ == "__main__":
    args = get_cli_args()
    model, vocab = main(args)
    test_model(model, vocab)