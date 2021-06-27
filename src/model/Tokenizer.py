from typing import List

from definitions import TOKENIZER_PATH

from transformers import MBart50TokenizerFast


class TokenizerWrapper:
    def __init__(self):
        self._tokenizer = MBart50TokenizerFast.from_pretrained(TOKENIZER_PATH)

        special_tokens = ['masc', 'fem', 'neut', 'undefined_g',
                          'past', 'pres', 'fut',
                          'sing', 'plur', 'undefined_n']

        self.special_tokens = {k: v for k, v in zip(special_tokens, self._tokenizer.additional_special_tokens)}

        self.bos_token_id = self._tokenizer.bos_token_id
        self.eos_token_id = self._tokenizer.eos_token_id

        self.vocab_size = property(lambda: len(self._tokenizer.get_vocab()))

    def encode(self, sentence: str, context: tuple):
        nsubj, gender, tense, number = context
        bos_token = self._tokenizer.bos_token
        eos_token = self._tokenizer.eos_token

        input = f'{nsubj} {self.special_tokens[gender]} ' \
                f'{self.special_tokens[tense]} {self.special_tokens[number]} ' \
                f'{bos_token} {sentence} {eos_token}'

        input = self._tokenizer(input, add_special_tokens=False, return_tensors='pt',
                                return_attention_mask=False, return_token_type_ids=False).input_ids

        return input

    def decode(self, output_ids: List[int]):
        decoded_output = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        return decoded_output
