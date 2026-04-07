from models.chmm import CHMMModel
from models.fhmm import FHMMModel
from models.finite_hmm import FiniteHMMModel
from models.hmm import HMMModel
from models.lstm import LSTMModel
from models.ngram import NGramModel
from models.switching_hmm import SwitchingHMMModel

MODEL_REGISTRY = {
    "hmm": HMMModel,
    "chmm": CHMMModel,
    "fhmm": FHMMModel,
    "f_hmm": FiniteHMMModel,
    "switching_hmm": SwitchingHMMModel,
    "lstm": LSTMModel,
    "n_gram": NGramModel,
}

__all__ = [
    "CHMMModel",
    "FHMMModel",
    "FiniteHMMModel",
    "HMMModel",
    "LSTMModel",
    "MODEL_REGISTRY",
    "NGramModel",
    "SwitchingHMMModel",
]