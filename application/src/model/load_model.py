from torch import load


def load_model(model, path):
    state_dict = load(path)
    model.load_state_dict(state_dict)