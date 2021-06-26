import torch
import torch.nn as nn


class Seq2SeqTransformer(nn.Module):
    def __init__(self, embedding_size, nhead,
                 num_encoder_layers, num_decoder_layers,
                 dim_feedforward, dropout, vocab_size,
                 max_seq_len, pad_token_id, device):
        super(Seq2SeqTransformer, self).__init__()

        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.embedding_size = embedding_size

        self.to(device)

        self.device = device

        self.word_embedding = nn.Embedding(vocab_size, embedding_size, pad_token_id)
        self.input_pos_encoding = nn.Embedding(max_seq_len, embedding_size)
        self.target_pos_encoding = nn.Embedding(max_seq_len, embedding_size)

        self.transformer = nn.Transformer(embedding_size, nhead, num_encoder_layers,
                                          num_decoder_layers, dim_feedforward, dropout)

        self.fc_out = nn.Linear(embedding_size, vocab_size)

    def get_padding_mask(self, input):
        # input shape: (seq_len, batch_size)
        padding_mask = input.permute(1, 0) == self.pad_token_id
        return padding_mask.to(self.device)

    def forward(self, input, target):
        # input shape: (input_seq_len, batch_size)
        # target shape: (target_seq_len, batch_size)

        embedded_input = self.word_embedding(input)
        embedded_target = self.word_embedding(target)
        # embedded_input shape: (input_seq_len, batch_size, embedding_size)
        # embedded_target shape: (target_seq_len, batch_size, embedding_size)

        batch_size = input.shape[1]

        input_seq_len = input.shape[0]
        target_seq_len = target.shape[0]

        input_positions = torch.arange(0, input_seq_len).unsqueeze(1).expand(input_seq_len, batch_size).to(self.device)
        target_positions = torch.arange(0, target_seq_len).unsqueeze(1).expand(target_seq_len, batch_size).to(
            self.device)
        # input_positions shape: (input_seq_len, batch_size)
        # target_positions shape: (target_seq_len, batch_size)

        input_positions = self.input_pos_encoding(input_positions)
        target_positions = self.target_pos_encoding(target_positions)
        # input_positions shape: (input_seq_len, batch_size, embedding_size)
        # target_positions shape: (target_seq_len, batch_size, embedding_size)

        embedded_input += input_positions
        embedded_target += target_positions

        input_padding_mask = self.get_padding_mask(input)
        target_padding_mask = self.get_padding_mask(target)
        # input_padding_mask shape: (batch_size, input_seq_len)

        target_mask = self.transformer.generate_square_subsequent_mask(target_seq_len).to(self.device)
        # target_mask shape: (target_seq_len, target_seq_len)

        output = self.transformer(embedded_input, embedded_target,
                                  tgt_mask=target_mask,
                                  src_key_padding_mask=input_padding_mask,
                                  tgt_key_padding_mask=target_padding_mask)
        # output shape: (target_seq_len, batch_size, embedding_size)

        output = self.fc_out(output)
        # output shape: (target_seq_len, batch_size, vocab_size)

        return output