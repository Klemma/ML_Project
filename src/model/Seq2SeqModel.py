import torch.nn as nn
import torch

from random import random

from src.model.ContextMem import ContextMem
from src.model.DecoderRNN import DecoderRNN
from src.model.EncoderRNN import EncoderRNN


class Seq2SeqModel(nn.Module):
    def __init__(self,
                 vocab_size, embedding_size, hidden_size, output_size,
                 gender_input_size, tense_input_size, context_hidden_size, context_output_size,
                 pad_idx, device, num_layers, dropout_p, pretrained_embedding=None):
        super(Seq2SeqModel, self).__init__()

        self.device = device
        self.num_layers = num_layers

        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, padding_idx=pad_idx)
            self.pretrained_embedding_loaded = True
        else:
            self.embedding = nn.Sequential(
                nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx),
                nn.Dropout(dropout_p)
            )
            self.pretrained_embedding_loaded = False

        self.context_mem = ContextMem(gender_input_size, tense_input_size, context_hidden_size, context_output_size,
                                      embedding_size, device).to(device)
        self.encoder = EncoderRNN(vocab_size, embedding_size, hidden_size,
                                  pad_idx, device, num_layers, dropout_p,
                                  self.embedding, self.pretrained_embedding_loaded).to(device)
        self.decoder = DecoderRNN(vocab_size, embedding_size, hidden_size, output_size,
                                  pad_idx, device, num_layers, dropout_p,
                                  self.embedding, self.pretrained_embedding_loaded).to(device)

        self.vocab_size = vocab_size

    def forward(self, input, target, context, teacher_forcing_ratio=0.5):
        batch_size = input.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.vocab_size

        outputs = torch.zeros(target_len, batch_size, target_vocab_size, device=self.device)

        nsubj, gender, tense = context
        # nsubj_shape:  (batch_size)
        # gender_shape: (batch_size, gender_input_size)
        # tense_shape:  (batch_size, tense_input_size)
        if self.pretrained_embedding_loaded:
            with torch.no_grad():
                nsubj_embedding = self.embedding(nsubj).squeeze(0)
        else:
            nsubj_embedding = self.embedding(nsubj).squeeze(0)
            # nsubj_embedding_shape: (batch_size, embedding_size)

        hidden = self.context_mem(nsubj_embedding, gender, tense)
        cell = hidden.clone()
        # hidden, cell shapes: (batch_size, context_output_size=hidden_size)

        hidden = torch.cat([hidden.unsqueeze(0)] * 2, 0)
        cell = torch.cat([cell.unsqueeze(0)] * 2, 0)
        # hidden, cell shapes: (2, batch_size, context_output_size=hidden_size)

        encoder_states, hidden, cell = self.encoder(input, hidden, cell)
        # hidden, cell shapes: (2, batch_size, hidden_size)
        # encoder_states_shape: (seq_len, batch_size, hidden_size * 2)

        prev_token_idx = target[0]
        # prev_token_shape: (batch_size)

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(prev_token_idx, encoder_states, hidden, cell)
            # output_shape: (1, batch_size, output_size)
            outputs[t] = output.squeeze(0)

            best_prediction = outputs[t].argmax(dim=1)
            # best_prediction_shape: (batch_size)
            prev_token_idx = target[t] if random() < teacher_forcing_ratio else best_prediction

        return outputs
