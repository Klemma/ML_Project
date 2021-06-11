import pickle
from src.vocab.Vocab import Vocab


class VocabUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        try:
            return super(VocabUnpickler, self).find_class(__name__, name)
        except AttributeError:
            return super(VocabUnpickler, self).find_class(module, name)


def load_vocab(path):
    with open(path, 'rb') as vocab_file:
        vocab = VocabUnpickler(vocab_file).load()

    return vocab
