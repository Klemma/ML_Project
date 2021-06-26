from typing import List

from transformers import BertTokenizerFast


class TokenizerWrapper:
    def __init__(self):
        self._tokenizer = BertTokenizerFast.from_pretrained('DeepPavlov/rubert-base-cased')
        self.special_tokens = {
            'bos_token': '[BOS]',
            'eos_token': '[EOS]',
            'masc': '[MASC_G]',
            'fem': '[FEM_G]',
            'neut': '[NEUT_G]',
            'undefined_g': '[UNDEF_G]',
            'past': '[PAST_T]',
            'pres': '[PRES_T]',
            'fut': '[FUT_T]',
            'sing': '[SING_N]',
            'plur': '[PLUR_N]',
            'undefined_n': '[UNDEF_N]'
        }
        self._tokenizer.add_special_tokens({'bos_token': self.special_tokens['bos_token'],
                                            'eos_token': self.special_tokens['eos_token'],
                                            'additional_special_tokens': list(self.special_tokens.values())[2:]})
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
                                return_attention_mask=False, return_token_type_ids=False).input_ids.permute(1, 0)

        return input

    def decode(self, output_ids: List[int]):
        decoded_output = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        return decoded_output
