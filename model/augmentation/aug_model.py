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
        self.model_type = args.model_type
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.dropout_rate = args.dropout_rate
        self.vocab_size = args.vocab_size
        self.latent_size = args.latent_size
        self.pad_id = args.pad_id
        self.unk_id = args.unk_id
        self.max_seq_len = args.max_seq_len
        self.variational = args.variational
        self.denosing_rate = args.denosing_rate

        self.embed = nn.Embedding(num_embeddings=self.vocab_size,
                                  embedding_dim=self.embed_size,
                                  padding_idx=self.pad_id)
        # Encoder & Decoder Layer
        if self.model_type == 'rnn':
            self.encoder = nn.RNN(input_size=self.embed_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=False, # Always False for decoder
                                  dropout=self.dropout_rate,
                                  batch_first=True)
            self.decoder = nn.RNN(input_size=self.embed_size + self.latent_size if self.variational else self.embed_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=False, # Always False for decoder
                                  dropout=self.dropout_rate,
                                  batch_first=True)
        elif self.model_type == 'gru':
            self.encoder = nn.GRU(input_size=self.embed_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=False,
                                  batch_first=True,
                                  dropout=self.dropout_rate)
            self.decoder = nn.GRU(input_size=self.embed_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=False, # Always False for decoder
                                  batch_first=True,
                                  dropout=self.dropout_rate)
        elif self.model_type == 'lstm':
            self.encoder = nn.LSTM(input_size=self.embed_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.num_layers,
                                   bidirectional=False,
                                   batch_first=True,
                                   dropout=self.dropout_rate)
            self.decoder = nn.LSTM(input_size=self.embed_size + self.latent_size if self.variational else self.embed_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.num_layers,
                                   bidirectional=False, # Always False for decoder
                                   batch_first=True,
                                   dropout=self.dropout_rate)
        elif self.model_type == 'transformer':
            enc_layer = nn.TransformerEncoderLayer(d_model=self.embed_size, nhead=8, batch_first=True)
            dec_layer = nn.TransformerDecoderLayer(d_model=self.embed_size, nhead=8, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer=enc_layer, num_layers=6)
            self.decoder = nn.TransformerDecoder(decoder_layer=dec_layer, num_layers=6)
            self.pos_embed = nn.Embedding(num_embeddings=self.max_seq_len, embedding_dim=self.embed_size)
        else:
            raise NotImplementedError(f'Not Implemented Model Type: {self.model_type}')

        # Autoencoder Layer
        if self.variational and self.model_type in ['rnn', 'gru', 'lstm']:
            self.variational_mu = nn.Linear(in_features=self.num_layers * self.hidden_size,
                                            out_features=self.latent_size)
            self.variational_logvar = nn.Linear(in_features=self.num_layers * self.hidden_size,
                                                out_features=self.latent_size)
            self.variational_hidden = nn.Linear(in_features=self.latent_size,
                                                out_features=self.num_layers * self.hidden_size,)
        elif self.variational and self.model_type == 'transformer':
            self.variational_mu = nn.Linear(in_features=self.embed_size,
                                            out_features=self.latent_size)
            self.variational_logvar = nn.Linear(in_features=self.embed_size,
                                                out_features=self.latent_size)
            self.variational_hidden = nn.Linear(in_features=self.latent_size,
                                                out_features=self.embed_size)

        # Output Layer
        self.output = nn.Linear(in_features=self.hidden_size if self.model_type in ['rnn', 'gru', 'lstm'] else self.embed_size,
                                out_features=self.vocab_size)

    def forward(self, input_seq:torch.Tensor) -> tuple: # (torch.Tensor, torch.Tensor, torch.Tensor)
        # input_seq: (batch_size, max_seq_len)
        if self.denosing_rate > 0:
            denosied_input_seq = input_seq[:, 1:-1].clone() # (batch_size, max_seq_len - 2) # <bos> and <eos> are not noised
            denosing_mask = torch.rand_like(denosied_input_seq, dtype=torch.float32) < self.denosing_rate
            denosied_input_seq[denosing_mask] = self.unk_id

            input_seq = torch.cat([input_seq[:, :1], denosied_input_seq, input_seq[:, -1:]], dim=-1) # (batch_size, max_seq_len)

        if self.model_type in ['rnn', 'gru', 'lstm']:
            # Encoder
            input_embed_encoder = self.embed(input_seq) # (batch_size, max_seq_len, embed_size)
            input_length = torch.sum(input_seq != self.pad_id, dim=-1) # (batch_size)
            sorted_length, sorted_idx = torch.sort(input_length, descending=True)
            sorted_encoder_input = input_embed_encoder[sorted_idx] # (batch_size, max_seq_len, embed_size)
            packed_encoder_input = pack_padded_sequence(sorted_encoder_input, sorted_length.data.tolist(), batch_first=True)

            _, encoder_hidden = self.encoder(packed_encoder_input) # (batch_size, max_seq_len, output_size), (num_layers, batch_size, hidden_size)

            # Autoencoder process
            if self.variational: # Variational Autoencoder
                enc_hidden = encoder_hidden[0] if self.model_type == 'lstm' else encoder_hidden # (num_layers, batch_size, hidden_size)
                enc_hidden = enc_hidden.view(-1, self.num_layers * self.hidden_size) # (batch_size, num_layers * hidden_size)
                mu = self.variational_mu(enc_hidden) # (batch_size, latent_size)
                logvar = self.variational_logvar(enc_hidden) # (batch_size, latent_size)
                # Reparameterization Trick
                std = torch.exp(0.5 * logvar) # (batch_size, latent_size)
                eps = torch.randn_like(std) # (batch_size, latent_size)
                noise = mu + eps * std # (batch_size, latent_size)
                # Noise to hidden state
                noise_hidden = self.variational_hidden(noise) # (batch_size, num_layers * hidden_size)
                encoder_hidden = encoder_hidden + noise_hidden.view(self.num_layers, -1, self.hidden_size) # (num_layers, batch_size, hidden_size)
                """
                #NOISE CONCAT
                noise = noise.unsqueeze(1).repeat(1, self.max_seq_len-1, 1) # (batch_size, max_seq_len-1, latent_size)
                # Define decoder input
                input_embed_decoder = self.embed(input_seq[:, :-1]) # (batch_size, max_seq_len-1, embed_size) # Remove <eos> token
                input_embed_decoder = torch.cat([input_embed_decoder, noise], dim=-1) # (batch_size, max_seq_len-1, embed_size + latent_size)
                """
                """
                #NOISE ADD
                noise = self.variational_hidden(noise) # (batch_size, embed_size)
                noise = noise.unsqueeze(1).repeat(1, self.max_seq_len-1, 1) # (batch_size, max_seq_len-1, embed_size)
                input_embed_decoder = self.embed(input_seq[:, :-1]) # (batch_size, max_seq_len-1, embed_size) # Remove <eos> token
                input_embed_decoder = input_embed_decoder + noise # (batch_size, max_seq_len-1, embed_size)
                """

                # Define decoder input
                input_embed_decoder = self.embed(input_seq[:, :-1]) # (batch_size, max_seq_len-1, embed_size) # Remove <eos> token
                input_length = input_length - 1 # (batch_size) # Remove <eos> token for decoder input
                sorted_length, sorted_idx = torch.sort(input_length, descending=True)
                sorted_decoder_input = input_embed_decoder[sorted_idx] # (batch_size, max_seq_len-1, embed_size)
                packed_decoder_input = pack_padded_sequence(sorted_decoder_input, sorted_length.data.tolist(), batch_first=True)
            else: # Deterministic Autoencoder
                mu = None
                logvar = None

                # Define decoder input
                input_embed_decoder = self.embed(input_seq[:, :-1]) # (batch_size, max_seq_len-1, embed_size) # Remove <eos> token
                input_length = input_length - 1 # (batch_size) # Remove <eos> token for decoder input
                sorted_length, sorted_idx = torch.sort(input_length, descending=True)
                sorted_decoder_input = input_embed_decoder[sorted_idx] # (batch_size, max_seq_len-1, embed_size)
                packed_decoder_input = pack_padded_sequence(sorted_decoder_input, sorted_length.data.tolist(), batch_first=True)

            decoder_output, _ = self.decoder(packed_decoder_input, encoder_hidden) # (batch_size, max_seq_len-1, output_size), (num_layers, batch_size, hidden_size)
            decoder_output, _ = pad_packed_sequence(decoder_output, batch_first=True, total_length=self.max_seq_len-1) # (batch_size, max_seq_len-1, output_size)
            _, reversed_idx = torch.sort(sorted_idx) # (batch_size)
            decoder_output = decoder_output[reversed_idx] # (batch_size, max_seq_len-1, output_size)

            decoder_output = self.output(decoder_output) # (batch_size, max_seq_len-1, vocab_size)
            decoder_output = decoder_output.view(-1, decoder_output.size(-1)) # (batch_size * max_seq_len-1, vocab_size)
            decoder_output = F.log_softmax(decoder_output, dim=-1) # (batch_size * max_seq_len-1, vocab_size)
        else: # Transformer
            # Encoder
            encoder_pos_ids = torch.arange(0, input_seq.size(1), dtype=torch.long, device=input_seq.device) # (max_seq_len)
            input_embed_encoder = self.embed(input_seq) # (batch_size, max_seq_len, embed_size)
            encoder_pos_embed = self.embed(encoder_pos_ids) # (max_seq_len, embed_size)
            encoder_input = input_embed_encoder + encoder_pos_embed # (batch_size, max_seq_len, embed_size)

            encoder_output = self.encoder(encoder_input) # (batch_size, max_seq_len, embed_size)

            # Autoencoder process
            if self.variational: # Variational Autoencoder
                enc_hidden = encoder_output.mean(dim=1) # (batch_size, embed_size)
                mu = self.variational_mu(enc_hidden) # (batch_size, latent_size)
                logvar = self.variational_logvar(enc_hidden) # (batch_size, latent_size)
                # Reparameterization Trick
                std = torch.exp(0.5 * logvar) # (batch_size, latent_size)
                eps = torch.randn_like(std) # (batch_size, latent_size)
                noise = mu + eps # (batch_size, latent_size)
                noise_embed = self.variational_hidden(noise) # (batch_size, embed_size)
                noise_embed = noise_embed.unsqueeze(1).repeat(1, self.max_seq_len-1, 1) # (batch_size, max_seq_len-1, embed_size)

                # Define decoder input
                decoder_pos_ids = torch.arange(0, input_seq[:, :-1].size(1), dtype=torch.long, device=input_seq.device) # (max_seq_len-1)
                input_embed_decoder = self.embed(input_seq[:, :-1]) # (batch_size, max_seq_len-1, embed_size) # Remove <eos> token
                decoder_pos_embed = self.embed(decoder_pos_ids) # (max_seq_len-1, embed_size)
                input_embed_decoder = input_embed_decoder + decoder_pos_embed + noise_embed # (batch_size, max_seq_len-1, embed_size)
            else: # Deterministic Autoencoder
                mu = None
                logvar = None

                # Define decoder input - Denosing: 50% of tokens are replaced by <unk> token
                decoder_pos_ids = torch.arange(0, input_seq[:, :-1].size(1), dtype=torch.long, device=input_seq.device) # (max_seq_len-1)
                input_embed_decoder = self.embed(input_seq[:, :-1]) # (batch_size, max_seq_len-1, embed_size)

                decoder_pos_embed = self.embed(decoder_pos_ids) # (max_seq_len-1, embed_size)
                input_embed_decoder = input_embed_decoder + decoder_pos_embed # (batch_size, max_seq_len-1, embed_size)

            # Decoder
            tgt_mask = self.generate_square_subsequent_mask(input_seq[:, :-1].size(1), device=input_seq.device) # (max_seq_len-1, max_seq_len-1)
            tgt_key_padding_mask = (input_seq[:, :-1] == self.pad_id) # (batch_size, max_seq_len-1)

            decoder_output = self.decoder(input_embed_decoder, encoder_output,
                                          tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask) # (batch_size, max_seq_len-1, embed_size)

            decoder_output = self.output(decoder_output) # (batch_size, max_seq_len-1, vocab_size)
            decoder_output = decoder_output.view(-1, decoder_output.size(-1)) # (batch_size * (max_seq_len-1), vocab_size)
            decoder_output = F.log_softmax(decoder_output, dim=-1) # (batch_size * (max_seq_len-1), vocab_size)

        return decoder_output, mu, logvar

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        mask = torch.tril(torch.ones(sz, sz, dtype=torch.float, device=device))
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask

def kl_annealing(epoch:int, max_epoch:int, start_val:float=0, end_val:float=1, annealing_rate:float=0.95) -> float:
    return end_val + (start_val-end_val) * annealing_rate ** epoch
