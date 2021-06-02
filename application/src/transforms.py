from typing import List

import nltk
import torch

from application.src.vocab.special_tokens import token_to_idx


def text_to_indices(text: str, vocab) -> List[int]:
    try:
        tokenized_text = nltk.tokenize.word_tokenize(text.lower(), language='russian')
    except LookupError:
        nltk.download('punkt')
        tokenized_text = nltk.tokenize.word_tokenize(text.lower(), language='russian')
    transformed = [vocab.token_to_idx(token) for token in tokenized_text]
    return transformed


def text_to_tensor(text: str, vocab, max_seq_len=8):
    transformed = text_to_indices(text, vocab)
    pad_idx = token_to_idx.get('<PAD>')
    sos_idx = token_to_idx.get('<SOS>')
    eos_idx = token_to_idx.get('<EOS>')

    pad_size = 0
    if len(transformed) >= max_seq_len:
        transformed = transformed[:max_seq_len]
    else:
        pad_size = max_seq_len - len(transformed)
        transformed.extend([pad_idx] * pad_size)
    transformed.insert(0, sos_idx)
    transformed.insert(len(transformed) - pad_size, eos_idx)

    tensor = torch.tensor(transformed, dtype=torch.long)
    return tensor.unsqueeze(0)


gender_idx_to_vector = {
    0: [0, 1, 0, 0],
    1: [0, 0, 1, 0],
    2: [1, 0, 0, 0],
    3: [0, 0, 1, 0]
}

tense_idx_to_vector = {
    0: [0, 1, 0, 0],
    1: [0, 0, 1, 0],
    2: [1, 0, 0, 0],
    3: [0, 0, 0, 1]
}

# gender_idx_to_label = {
#     0: 'masc',
#     1: 'neut',
#     2: 'fem',
#     3: 'undefined'
# }
#
# tense_idx_to_label = {
#     0: 'past',
#     1: 'pres',
#     2: 'fut',
#     3: 'undefined'
# }
