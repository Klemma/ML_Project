import torch
import torch.nn as nn


class ContextMem(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nsubj_embedding_size, device):
        super(ContextMem, self).__init__()

        self.device = device

        self.gender_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.tense_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.fc_out = nn.Linear(hidden_size * 2 + nsubj_embedding_size, output_size, bias=False)

    def forward(self, nsubj_embedding, gender, tense):
        # nsubj_embedding_shape: (batch_size, embedding_size)
        # gender_shape: (batch_size, input_size)
        # tense_shape: (batch_size, input_size)

        gender = self.gender_proj(gender)
        # gender_shape: (batch_size, hidden_size)

        tense = self.tense_proj(tense)
        # tense_shape: (batch_size, hidden_size)

        context = torch.cat([nsubj_embedding, gender, tense], dim=1)
        # context_shape: (batch_size, hidden_size * 2 + embedding_size)

        context = self.fc_out(context)
        # context_shape: (batch_size, output_size)

        return context
