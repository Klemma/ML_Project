import torch.nn as nn


class DecoderRNN(nn.Module):
    def __init__(self,
                 vocab_size, embedding_size, hidden_size, output_size,
                 pad_idx, device, num_layers, dropout_p):
        super(DecoderRNN, self).__init__()

        self.device = device

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, embedding_size, pad_idx),
            nn.Dropout(dropout_p)
        )
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        # x_shape:      (seq_len=1, batch_size)
        # hidden_shape: (num_layers, batch_size, hidden_size)
        # cell_shape:   (num_layers, batch_size, hidden_size)

        embedding = self.embedding(x)
        # embedding_shape: (seq_len=1, batch_size, embedding_size)

        lstm_out, (hidden, cell) = self.lstm(embedding, (hidden, cell))
        # lstm_out_shape: (seq_len=1, batch_size, hidden_size)

        fc_out = self.fc(lstm_out)
        # fc_out_shape: (seq_len=1, batch_size, output_size)
        # output_shape: (seq_len=1, batch_size, output_size)

        return fc_out, hidden, cell
