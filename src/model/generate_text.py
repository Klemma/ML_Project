from typing import List

import application.src.transforms as transforms
from application.src.vocab.special_tokens import token_to_idx

import torch


def generate_sentence(model, sentence: str, context: List[str], lemm_vocab, orig_vocab, device=torch.device('cpu'), max_seq_len=8):
    with torch.no_grad():
        model.eval()

        nsubj, gender, tense = context

        nsubj = torch.tensor([orig_vocab.token_to_idx(nsubj)], device=device).unsqueeze(0)
        gender = torch.tensor([transforms.gender_idx_to_vector.get(gender)], dtype=torch.float32, device=device)
        tense = torch.tensor([transforms.tense_idx_to_vector.get(tense)], dtype=torch.float32, device=device)

        nsubj_embedding = model.decoder.embedding(nsubj).squeeze(0)

        hidden = model.context_mem(nsubj_embedding, gender, tense)
        cell = hidden.clone()

        if model.num_layers == 1:
            hidden.unsqueeze_(0)
            cell.unsqueeze_(0)
            # hidden, cell shapes: (1, batch_size, context_output_size=hidden_size)
        else:
            hidden = torch.cat([hidden.unsqueeze(0)] * model.num_layers, 0)
            cell = torch.cat([cell.unsqueeze(0)] * model.num_layers, 0)
            # hidden, cell shapes: (num_layers, batch_size, context_output_size=hidden_size)

        input_tensor = transforms.text_to_tensor(sentence, lemm_vocab, max_seq_len).to(device)
        input_tensor = torch.transpose(input_tensor, 1, 0)
        sos_idx = token_to_idx.get('<SOS>')
        eos_idx = token_to_idx.get('<EOS>')

        hidden, cell = model.encoder(input_tensor, hidden, cell)

        predicted_indexes = [sos_idx]

        for _ in range(1, max_seq_len):
            prev_idx = torch.tensor([predicted_indexes[-1]], dtype=torch.long, device=device)

            output, hidden, cell = model.decoder(prev_idx, hidden, cell)
            output = output.squeeze(0)

            best_prediction = output.argmax(dim=1).item()

            if best_prediction == eos_idx:
                break

            predicted_indexes.append(best_prediction)

    predicted_tokens = [orig_vocab.idx_to_token(idx) for idx in predicted_indexes]

    generated_text = ''
    for token in predicted_tokens[1:]:
        generated_text += token + ' '

    return generated_text[:-1]
