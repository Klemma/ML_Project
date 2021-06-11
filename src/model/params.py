from src.vocab.special_tokens import token_to_idx

params = {
    'vocab_size': 125000,
    'embedding_size': 300,
    'hidden_size': 1024,
    'output_size': 125000,
    'gender_input_size': 4,
    'tense_input_size': 3,
    'context_hidden_size': 512,
    'context_output_size': 1024,
    'pad_idx': token_to_idx.get('<PAD>'),
    'device': 'cpu',
    'num_layers': 1,
    'dropout_p': 0.5
}
