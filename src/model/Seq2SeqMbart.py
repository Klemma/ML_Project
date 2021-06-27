from definitions import MODEL_PATH

from transformers import MBartForConditionalGeneration


class Seq2SeqMbart:
    def __init__(self):
        self._model: MBartForConditionalGeneration = MBartForConditionalGeneration.from_pretrained(MODEL_PATH)

    def generate(self, input_ids, bos_token_id, eos_token_id):
        output = self._model.generate(input_ids,
                                      forced_bos_token_id=bos_token_id,
                                      forced_eos_token_id=eos_token_id)
        return output[0]
