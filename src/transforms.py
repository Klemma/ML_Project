from typing import List

import nltk
import torch

from src.vocab.special_tokens import token_to_idx


def text_to_indices(text: str, vocab) -> List[int]:
    try:
        tokenized_text = nltk.tokenize.word_tokenize(text.lower(), language='russian')
    except LookupError:
        nltk.download('punkt')
        tokenized_text = nltk.tokenize.word_tokenize(text.lower(), language='russian')
    transformed = [vocab.token_to_idx(token) for token in tokenized_text]
    return transformed


def text_to_tensor(text: str, vocab, max_seq_len=40) -> torch.tensor:
    transformed_text = text_to_indices(text, vocab)
    pad_idx = token_to_idx.get('<pad>')
    sos_idx = token_to_idx.get('<sos>')
    eos_idx = token_to_idx.get('<eos>')

    pad_size = 0
    if len(transformed_text) >= max_seq_len:
        transformed_text = transformed_text[:max_seq_len]
    else:
        pad_size = max_seq_len - len(transformed_text)
        transformed_text.extend([pad_idx] * pad_size)
    transformed_text.insert(0, sos_idx)
    transformed_text.insert(len(transformed_text) - pad_size, eos_idx)

    tensor = torch.tensor(transformed_text, dtype=torch.long)
    return tensor.unsqueeze(0)


gender_to_vec = {
    0: [1, 0, 0, 0],
    1: [0, 1, 0, 0],
    2: [0, 0, 1, 0],
    3: [0, 0, 0, 1]
}

tense_to_vec = {
    0: [1, 0, 0],
    1: [0, 1, 0],
    2: [0, 0, 1]
}
