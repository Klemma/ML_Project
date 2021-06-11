import torch.nn as nn
import torch

from .ContextMem import ContextMem
from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN

from random import random


class Seq2SeqModel(nn.Module):
    def __init__(self,
                 encoder_vocab_size, decoder_vocab_size,
                 embedding_size, hidden_size, output_size,
                 context_input_size, context_hidden_size, context_output_size,
                 pad_idx, device, num_layers, dropout_p):

        super(Seq2SeqModel, self).__init__()

        self.device = device

        self.num_layers = num_layers
        self.decoder_vocab_size = decoder_vocab_size

        self.context_mem = ContextMem(context_input_size, context_hidden_size, context_output_size, embedding_size,
                                      device).to(device)
        self.encoder = EncoderRNN(encoder_vocab_size, embedding_size, hidden_size, pad_idx, device, num_layers,
                                  dropout_p).to(device)
        self.decoder = DecoderRNN(decoder_vocab_size, embedding_size, hidden_size, output_size, pad_idx, device,
                                  num_layers, dropout_p).to(device)

    def forward(self, input, target, context, teacher_forcing_ratio=0.5):
        batch_size = input.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.decoder_vocab_size

        outputs = torch.zeros(target_len, batch_size, target_vocab_size, device=self.device)

        nsubj, gender, tense = context
        # nsubj_shape:  (batch_size)
        # gender_shape: (batch_size, context_input_size)
        # tense_shape:  (batch_size, context_input_size)

        nsubj_embedding = self.decoder.embedding(nsubj).squeeze(0)
        # nsubj_embedding_shape: (batch_size, embedding_size)

        hidden = self.context_mem(nsubj_embedding, gender, tense)
        cell = hidden.clone()
        # hidden, cell shapes: (batch_size, context_output_size=hidden_size)

        if self.num_layers == 1:
            hidden.unsqueeze_(0)
            cell.unsqueeze_(0)
            # hidden, cell shapes: (1, batch_size, context_output_size=hidden_size)
        else:
            hidden = torch.cat([hidden.unsqueeze(0)] * self.num_layers, 0)
            cell = torch.cat([cell.unsqueeze(0)] * self.num_layers, 0)
            # hidden, cell shapes: (num_layers, batch_size, context_output_size=hidden_size)

        hidden, cell = self.encoder(input, hidden, cell)
        # hidden, cell shapes: (num_layers, batch_size, hidden_size)

        prev_token_idx = target[0]
        # prev_token_shape: (batch_size)

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(prev_token_idx, hidden, cell)
            outputs[t] = output.squeeze(0)

            best_prediction = outputs[t].argmax(dim=1)
            # best_prediction_shape: (batch_size)
            prev_token_idx = target[t] if random() < teacher_forcing_ratio else best_prediction

        return outputs
