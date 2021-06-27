import torch


def delemmatize_sentence(model, tokenizer, sentence: str, context: tuple) -> str:
    device = torch.device('cpu')

    encoded_sentence = tokenizer.encode(sentence, context).to(device)
    generated_ids = model.generate(encoded_sentence, tokenizer.bos_token_id, tokenizer.eos_token_id)

    decoded_output = tokenizer.decode(generated_ids)
    return decoded_output
