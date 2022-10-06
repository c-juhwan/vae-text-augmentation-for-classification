# standara Library Modules
import argparse
from unicodedata import bidirectional
# Pytorch Modules
import torch
import torch.nn as nn
import torch.nn.functional as F

class AugmentationModel(nn.Module):
    def __init__(self, args:argparse.Namespace) -> None:
        super(AugmentationModel, self).__init__()
        self.model_type = args.model_type
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.dropout_rate = args.dropout_rate
        self.vocab_size = args.vocab_size
        self.pad_id = args.pad_id
        self.max_seq_len = args.aux_max_seq_len
        self.bidirectional = args.rnn_bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        self.embed = nn.Embedding(num_embeddings=self.vocab_size,
                                  embedding_dim=self.embed_size,
                                  padding_idx=self.pad_id)
        # Encoder & Decoder Layer
        if self.model_type == 'GRU':
            self.encoder = nn.GRU(input_size=self.embed_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=self.rnn_bidirectional,
                                  batch_first=True,
                                  dropout=self.dropout_rate)
            self.decoder = nn.GRU(input_size=self.embed_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=False, # Always False for decoder
                                  batch_first=True,
                                  dropout=self.dropout_rate)
        else:
            raise NotImplementedError(f'Not implemented model type: {self.model_type} for {args.task} task')

        # Variational Autoencoder Layer
        self.variational_mu = nn.Linear(in_features=self.num_directions * self.num_layers * self.hidden_size,
                                        out_features=self.latent_size)
        self.variational_logvar = nn.Linear(in_features=self.num_directions * self.num_layers * self.hidden_size,
                                            out_features=self.latent_size)
        self.variational_z = nn.Linear(in_features=self.latent_size,
                                       out_features=self.embed_size)

        # Output Layer
        self.output = nn.Linear(in_features=self.num_directions * self.hidden_size,
                                out_features=self.vocab_size)

    def forward(self, input_seq:torch.Tensor) -> tuple: # (torch.Tensor, torch.Tensor)
        # input_seq: (batch_size, max_seq_len)
        input_embed = self.embed(input_seq) # (batch_size, max_seq_len, embed_size)
        encoder_output, encoder_hidden = self.encoder(input_embed) # (batch_size, max_seq_len, output_size), (num_layers * num_directions, batch_size, hidden_size)

