from definitions import MODEL_PATH
from torch import load


def load_model(model):
    state_dict = load(MODEL_PATH)
    model.load_state_dict(state_dict)
