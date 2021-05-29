import pickle

from application.src.vocab.Vocab import Vocab


def load_vocabs(path):
    lemm_vocab, orig_vocab = Vocab(), Vocab()
    with open(path, 'rb') as vocabs_file:
        vocabs = pickle.load(vocabs_file)

    return vocabs
