import torch
import torch.nn as nn


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, output_size: int, pad_idx: int,
                 device, num_layers, dropout_p: float, embedding=None, pretrained_embedding_loaded=False):
        super(DecoderRNN, self).__init__()

        self.device = device

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = embedding
        self.pretrained_embedding_loaded = pretrained_embedding_loaded

        self.attn_weights = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
            nn.Softmax(dim=1)
        )
        self.lstm = nn.LSTM(embedding_size + 2 * hidden_size, hidden_size, num_layers, dropout=0.0)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)
        # x_shape: (seq_len=1, batch_size)
        # hidden_shape: (1, batch_size, hidden_size)
        # cell_shape: (1, batch_size, hidden_size)
        encoder_states = torch.transpose(encoder_states, 1, 0)
        # encoder_states_shape: (batch_size, seq_len, hidden_size * 2)
        if self.pretrained_embedding_loaded:
            with torch.no_grad():
                embedding = self.embedding(x)
        else:
            embedding = self.embedding(x)
        # embedding_shape: (seq_len=1, batch_size, embedding_size)

        seq_len = encoder_states.shape[1]
        hidden_repeated = hidden.repeat(seq_len, 1, 1).permute(1, 0, 2)
        # hidden_repeated_shape: (batch_size, seq_len, hidden_size)

        attn_weights = self.attn_weights(torch.cat((hidden_repeated, encoder_states), dim=2))
        # attn_weights_shape: (batch_size, seq_len, 1)

        context_vec = torch.bmm(attn_weights.permute(0, 2, 1), encoder_states).permute(1, 0, 2)
        # context_vec_shape: (1, batch_size, hidden_size * 2)

        combined = torch.cat((embedding, context_vec), dim=2)
        # combined_shape: (1, batch_size, embedding_size + 2 * hidden_size)

        lstm_out, (hidden, cell) = self.lstm(combined, (hidden, cell))
        # lstm_out_shape: (seq_len=1, batch_size, hidden_size)
        fc_out = self.fc_out(lstm_out)
        # fc_out_shape: (seq_len=1, batch_size, output_size)

        return fc_out, hidden, cell
