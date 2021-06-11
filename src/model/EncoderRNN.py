import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self,
                 vocab_size, embedding_size, hidden_size,
                 pad_idx, device, num_layers, dropout_p):
        super(EncoderRNN, self).__init__()

        self.device = device
        self.num_layers = num_layers

        self.hidden_size = hidden_size

        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, embedding_size, pad_idx),
            nn.Dropout(dropout_p)
        )
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_p)

    def forward(self, x, hidden, cell):
        # x_shape: (seq_len, batch_size)
        embedding = self.embedding(x)
        # embedding_shape: (seq_len, batch_size, embedding_size)
        output, (hidden, cell) = self.lstm(embedding, (hidden, cell))
        # output_shape: (seq_len, batch_size, hidden_size)
        # hidden_shape: (num_layers, batch_size, hidden_size)
        # cell_shape: (num_layers, batch_size, hidden_size)
        return hidden, cell
