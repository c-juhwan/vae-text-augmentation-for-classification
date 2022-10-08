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
                                       out_features=self.num_directions * self.num_layers * self.hidden_size)

        # Output Layer
        self.output = nn.Linear(in_features=self.num_directions * self.hidden_size,
                                out_features=self.vocab_size)

    def forward(self, input_seq:torch.Tensor) -> tuple: # (torch.Tensor, torch.Tensor, torch.Tensor)
        # input_seq: (batch_size, max_seq_len)
        input_embed = self.embed(input_seq) # (batch_size, max_seq_len, embed_size)
        encoder_output, encoder_hidden = self.encoder(input_embed) # (batch_size, max_seq_len, output_size), (num_layers * num_directions, batch_size, hidden_size)

        # Variational Autoencoder
        encoder_hidden = encoder_hidden.view(-1, self.num_directions * self.num_layers * self.hidden_size) # (batch_size, num_directions * num_layers * hidden_size)
        mu = self.variational_mu(encoder_hidden) # (batch_size, latent_size)
        logvar = self.variational_logvar(encoder_hidden) # (batch_size, latent_size)
        # Reparameterization Trick
        std = torch.exp(0.5 * logvar) # (batch_size, latent_size)
        eps = torch.randn_like(std) # (batch_size, latent_size)
        z = mu + eps * std # (batch_size, latent_size)
        z = self.variational_z(z) # (batch_size, self.num_directions * self.num_layers * self.hidden_size)
        z = z.view(self.num_directions * self.num_layers, -1, self.hidden_size) # (num_direction * num_layers, batch_size, hidden_size)

        # Decoder
        input_embed_decoder = input_embed[:, :-1, :] # (batch_size, max_seq_len - 1, embed_size) # Remove <eos> token
        decoder_output, decoder_hidden = self.decoder(input_embed_decoder, z) # (batch_size, max_seq_len, output_size), (num_layers * num_directions, batch_size, hidden_size)
        decoder_output = self.output(decoder_output) # (batch_size, max_seq_len, vocab_size)
        decoder_output = decoder_output.view(-1, decoder_output.size(-1)) # (batch_size * max_seq_len, vocab_size)
        decoder_output = F.log_softmax(decoder_output, dim=-1) # (batch_size * max_seq_len, vocab_size)

        return decoder_output, mu, logvar

class GaussianKLLoss(nn.Module):
    def __init__(self, target_mu, target_logvar, device):
        super(GaussianKLLoss, self).__init__()

    def forward(self, mu1, logvar1, mu2, logvar2):
        numerator = logvar1.exp() + torch.pow(mu1 - mu2, 2)
        fraction = torch.div(numerator, (logvar2.exp()))
        kl = 0.5 * torch.sum(logvar2 - logvar1 + fraction - 1, dim=0)
        return kl.mean()
