import torch

params = {
    'embedding_size': 512,
    'nhead': 8,
    'num_encoder_layers': 4,
    'num_decoder_layers': 4,
    'dim_feedforward': 2048,
    'dropout': 0.1,
    'vocab_size': 119559,
    'max_seq_len': 110,
    'pad_token_id': 0,
    'device': torch.device('cpu')
}
