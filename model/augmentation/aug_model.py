# standara Library Modules
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AugmentationModel(nn.Module):
    def __init__(self, args:argparse.Namespace) -> None:
        super(AugmentationModel, self).__init__()
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.dropout_rate = args.dropout_rate
        self.vocab_size = args.vocab_size
        self.latent_size = args.latent_size
        self.pad_id = args.pad_id
        self.max_seq_len = args.max_seq_len
        self.variational = args.variational

        self.embed = nn.Embedding(num_embeddings=self.vocab_size,
                                  embedding_dim=self.embed_size,
                                  padding_idx=self.pad_id)
        # Encoder & Decoder Layer
        self.encoder = nn.GRU(input_size=self.embed_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=False,
                              batch_first=True,
                              dropout=self.dropout_rate)
        self.decoder = nn.GRU(input_size=self.embed_size + self.latent_size if self.variational else self.embed_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=False, # Always False for decoder
                              batch_first=True,
                              dropout=self.dropout_rate)

        # Autoencoder Layer
        if self.variational:
            self.variational_mu = nn.Linear(in_features=self.num_layers * self.hidden_size,
                                            out_features=self.latent_size)
            self.variational_logvar = nn.Linear(in_features=self.num_layers * self.hidden_size,
                                                out_features=self.latent_size)

        # Output Layer
        self.output = nn.Linear(in_features=self.hidden_size,
                                out_features=self.vocab_size)

    def forward(self, input_seq:torch.Tensor) -> tuple: # (torch.Tensor, torch.Tensor, torch.Tensor)
        # input_seq: (batch_size, max_seq_len)
        input_embed = self.embed(input_seq) # (batch_size, max_seq_len, embed_size)

        # Encoder
        input_length = torch.sum(input_seq != self.pad_id, dim=-1) # (batch_size)
        sorted_length, sorted_idx = torch.sort(input_length, descending=True)
        sorted_encoder_input = input_embed[sorted_idx] # (batch_size, max_seq_len, embed_size)
        packed_encoder_input = pack_padded_sequence(sorted_encoder_input, sorted_length.data.tolist(), batch_first=True)

        _, encoder_hidden = self.encoder(packed_encoder_input) # (batch_size, max_seq_len, output_size), (num_layers, batch_size, hidden_size)

        # Autoencoder process
        if self.variational: # Variational Autoencoder
            enc_hidden = encoder_hidden.view(-1, self.num_layers * self.hidden_size) # (batch_size, num_layers * hidden_size)
            mu = self.variational_mu(enc_hidden) # (batch_size, latent_size)
            logvar = self.variational_logvar(enc_hidden) # (batch_size, latent_size)
            # Reparameterization Trick
            std = torch.exp(0.5 * logvar) # (batch_size, latent_size)
            eps = torch.randn_like(std) # (batch_size, latent_size)
            noise = mu + eps * std # (batch_size, latent_size)

            # Define decoder input
            input_embed_decoder = input_embed[:, :-1, :] # (batch_size, max_seq_len - 1, embed_size) # Remove <eos> token
            input_embed_decoder = torch.cat([input_embed_decoder, noise.unsqueeze(1).repeat(1, self.max_seq_len - 1, 1)], dim=-1) # (batch_size, max_seq_len - 1, embed_size + latent_size)
            input_length = input_length - 1 # (batch_size) # Remove <eos> token for decoder input
            sorted_length, sorted_idx = torch.sort(input_length, descending=True)
            sorted_decoder_input = input_embed_decoder[sorted_idx] # (batch_size, max_seq_len - 1, embed_size)
            packed_decoder_input = pack_padded_sequence(sorted_decoder_input, sorted_length.data.tolist(), batch_first=True)
        else: # Deterministic Autoencoder
            mu = None
            logvar = None

            # Define decoder input
            input_embed_decoder = input_embed[:, :-1, :] # (batch_size, max_seq_len - 1, embed_size) # Remove <eos> token
            input_length = input_length - 1 # (batch_size) # Remove <eos> token for decoder input
            sorted_length, sorted_idx = torch.sort(input_length, descending=True)
            sorted_decoder_input = input_embed_decoder[sorted_idx] # (batch_size, max_seq_len - 1, embed_size)
            packed_decoder_input = pack_padded_sequence(sorted_decoder_input, sorted_length.data.tolist(), batch_first=True)

        decoder_output, _ = self.decoder(packed_decoder_input, encoder_hidden) # (batch_size, max_seq_len, output_size), (num_layers, batch_size, hidden_size)
        decoder_output, _ = pad_packed_sequence(decoder_output, batch_first=True, total_length=self.max_seq_len-1) # (batch_size, max_seq_len, output_size)
        _, reversed_idx = torch.sort(sorted_idx) # (batch_size)
        decoder_output = decoder_output[reversed_idx] # (batch_size, max_seq_len, output_size)

        decoder_output = self.output(decoder_output) # (batch_size, max_seq_len, vocab_size)
        decoder_output = decoder_output.view(-1, decoder_output.size(-1)) # (batch_size * max_seq_len, vocab_size)
        decoder_output = F.log_softmax(decoder_output, dim=-1) # (batch_size * max_seq_len, vocab_size)

        return decoder_output, mu, logvar

def kl_annealing(epoch:int, max_epoch:int, start_val:float=0, end_val:float=1, annealing_rate:float=0.95) -> float:
    return end_val + (start_val-end_val) * annealing_rate ** epoch
