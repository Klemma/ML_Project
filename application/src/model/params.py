from application.src.vocab.special_tokens import token_to_idx

params = {
    'encoder_vocab_size': 35000,
    'decoder_vocab_size': 75000,
    'embedding_size': 300,
    'hidden_size': 512,
    'output_size': 75000,
    'context_input_size': 4,
    'context_hidden_size': 256,
    'context_output_size': 512,
    'pad_idx': token_to_idx.get('<PAD>'),
    'device': 'cpu',
    'num_layers': 2,
    'dropout_p': 0.5
}
# output_size == decoder_vocab_size
# context_output_size == hidden_size
