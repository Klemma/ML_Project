from typing import Tuple

import torch

from src.transforms import gender_to_vec, tense_to_vec, text_to_tensor
from src.vocab.special_tokens import token_to_idx


def generate_sentence(model, sentence: str, context: Tuple[str, int, int], vocab, device=torch.device('cpu'),
                      max_seq_len=40):
    with torch.no_grad():
        model.eval()

        nsubj, gender, tense = context

        nsubj = torch.tensor([vocab.token_to_idx(nsubj)], device=device).unsqueeze(0)
        gender = torch.tensor([gender_to_vec[gender]], dtype=torch.float32, device=device)
        tense = torch.tensor([tense_to_vec[tense]], dtype=torch.float32, device=device)

        nsubj_embedding = model.decoder.embedding(nsubj).squeeze(0)

        hidden = model.context_mem(nsubj_embedding, gender, tense)
        cell = hidden.clone()

        hidden = torch.cat([hidden.unsqueeze(0)] * 2, 0)
        cell = torch.cat([cell.unsqueeze(0)] * 2, 0)
        # hidden, cell shapes: (2, batch_size, context_output_size=hidden_size)

        input_tensor = text_to_tensor(sentence, vocab, max_seq_len).to(device)
        input_tensor = torch.transpose(input_tensor, 1, 0)
        sos_idx = token_to_idx.get('<sos>')
        eos_idx = token_to_idx.get('<eos>')

        encoder_states, hidden, cell = model.encoder(input_tensor, hidden, cell)

        predicted_indexes = [sos_idx]

        for _ in range(1, max_seq_len):
            prev_idx = torch.tensor([predicted_indexes[-1]], dtype=torch.long, device=device)

            output, hidden, cell = model.decoder(prev_idx, encoder_states, hidden, cell)
            output = output.squeeze(0)

            best_prediction = output.argmax(dim=1).item()

            if best_prediction == eos_idx:
                break

            predicted_indexes.append(best_prediction)

    predicted_tokens = [vocab.idx_to_token(idx) for idx in predicted_indexes]
    return predicted_tokens[1:]
