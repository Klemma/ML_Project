from typing import List


class Vocab:
    def __init__(self, tokens: List[str] = None, unk_idx: int = None):
        self._tokens = tokens
        self._token_to_idx = {token: idx for idx, token in enumerate(tokens)} if tokens is not None else None
        self._unk_idx = unk_idx

    def token_to_idx(self, token: str) -> int:
        return self._token_to_idx.get(token, self._unk_idx)

    def idx_to_token(self, idx: int) -> str:
        return self._tokens[idx]
