import torch

from src.model.params import params


def delemmatize_sentence(model, tokenizer, sentence: str, context: tuple) -> str:
    predictions = [tokenizer.bos_token_id]
    max_seq_len = params.get('max_seq_len')
    device = params.get('device')
    encoded_sentence = tokenizer.encode(sentence, context).to(device)

    for i in range(max_seq_len):
        target = torch.tensor(predictions, device=device).unsqueeze(1)

        output = model(encoded_sentence, target)
        best_prediction = output.argmax(2)[-1].item()

        predictions.append(best_prediction)

        if best_prediction == tokenizer.eos_token_id:
            break

    decoded_output = tokenizer.decode(predictions)
    return decoded_output
