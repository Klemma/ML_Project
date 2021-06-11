import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, pad_idx: int,
                 device, num_layers, dropout_p: float, embedding=None, pretrained_embedding_loaded=False):
        super(EncoderRNN, self).__init__()

        self.device = device
        self.num_layers = num_layers

        self.hidden_size = hidden_size

        self.embedding = embedding
        self.pretrained_embedding_loaded = pretrained_embedding_loaded

        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=0.0, bidirectional=True)
        self.fc_compressor_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_compressor_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x, hidden, cell):
        # x_shape: (seq_len, batch_size)
        if self.pretrained_embedding_loaded:
            with torch.no_grad():
                embedding = self.embedding(x)
        else:
            embedding = self.embedding(x)
        # embedding_shape: (seq_len, batch_size, embedding_size)
        encoder_states, (hidden, cell) = self.lstm(embedding, (hidden, cell))
        # encoder_states: (seq_len, batch_size, hidden_size * 2)
        # hidden_shape: (num_layers=1 * 2, batch_size, hidden_size)
        # cell_shape: (num_layers=1 * 2, batch_size, hidden_size)

        bi_hidden = torch.cat((hidden[0], hidden[1]), dim=1).unsqueeze(0).permute(1, 0, 2)
        bi_cell = torch.cat((cell[0], cell[1]), dim=1).unsqueeze(0).permute(1, 0, 2)
        # bi_hidden, bi_cell shapes: (batch_size, 1, hidden_size * 2)

        hidden_compressed = self.fc_compressor_hidden(bi_hidden).permute(1, 0, 2)
        cell_compressed = self.fc_compressor_hidden(bi_cell).permute(1, 0, 2)
        # hidden_compressed, cell_compressed shapes: (1, batch_size, hidden_size)

        return encoder_states, hidden_compressed, cell_compressed
