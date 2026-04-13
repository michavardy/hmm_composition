#from models.chmm import CHMMModel
#from models.fhmm import FHMMModel
#from models.finite_hmm import FiniteHMMModel
from models.hmm import HMMModel
#from models.lstm import LSTMModel
#from models.ngram import NGramModel
from models.switching_hmm import SwitchingHMM

MODEL_REGISTRY = {
    "hmm": HMMModel,
    #"chmm": CHMMModel,
    #"fhmm": FHMMModel,
    #"f_hmm": FiniteHMMModel,
    "switching_hmm": SwitchingHMM,
    #"lstm": LSTMModel,
    #"n_gram": NGramModel,
}

