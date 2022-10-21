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
        self.bos_id = args.bos_id
        self.pad_id = args.pad_id
        self.unk_id = args.unk_id
        self.max_seq_len = args.max_seq_len
        self.variational_type = args.variational_type
        self.denosing_rate = args.denosing_rate
        self.activation_func = args.activation_func
        self.num_classes = args.num_classes

        # Embedding layer
        self.seq_embed = nn.Sequential(
            nn.Embedding(self.vocab_size, self.embed_size, padding_idx=self.pad_id),
            nn.Dropout(self.dropout_rate)
        )
        self.label_embed = nn.Sequential(
            nn.Embedding(self.num_classes, self.embed_size),
            nn.Dropout(self.dropout_rate)
        )
        self.pos_embed = nn.Sequential(
            nn.Embedding(self.max_seq_len, self.embed_size),
            nn.Dropout(self.dropout_rate)
        )

        # Encoder & Decoder Layer
        if self.model_type == 'rnn':
            self.encoder = nn.RNN(input_size=self.embed_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=False,
                                  dropout=self.dropout_rate,
                                  batch_first=True)
            self.decoder = nn.RNN(input_size=self.embed_size,
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
        elif self.model_type == 'transformer':
            enc_layer = nn.TransformerEncoderLayer(d_model=self.embed_size, nhead=8, batch_first=True)
            dec_layer = nn.TransformerDecoderLayer(d_model=self.embed_size, nhead=8, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer=enc_layer, num_layers=6)
            self.decoder = nn.TransformerDecoder(decoder_layer=dec_layer, num_layers=6)
        else:
            raise NotImplementedError(f'Not Implemented Model Type: {self.model_type}')

        # Autoencoder Layer
        if self.variational_type in ['VAE', 'CVAE'] and self.model_type in ['rnn', 'gru']:
            self.variational_mu = nn.Linear(in_features=self.num_layers * self.hidden_size,
                                            out_features=self.latent_size)
            self.variational_logvar = nn.Linear(in_features=self.num_layers * self.hidden_size,
                                                out_features=self.latent_size)
            self.variational_hidden = nn.Linear(in_features=self.latent_size,
                                                out_features=self.num_layers * self.hidden_size)
        elif self.variational_type in ['VAE', 'CVAE'] and self.model_type == 'transformer':
            self.variational_mu = nn.Linear(in_features=self.embed_size,
                                            out_features=self.latent_size)
            self.variational_logvar = nn.Linear(in_features=self.embed_size,
                                                out_features=self.latent_size)
            self.variational_hidden = nn.Linear(in_features=self.latent_size,
                                                out_features=self.embed_size)

        # Activation & Output Layer
        if self.activation_func == 'relu':
            self.activation = nn.ReLU()
        elif self.activation_func == 'tanh':
            self.activation = nn.Tanh()
        elif self.activation_func == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif self.activation_func == 'gelu':
            self.activation = nn.GELU()

        self.output = nn.Sequential(
            nn.Linear(in_features=self.hidden_size if self.model_type in ['rnn', 'gru'] else self.embed_size,
                      out_features=self.hidden_size * 2),
            self.activation,
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(in_features=self.hidden_size * 2,
                      out_features=self.vocab_size)
        )

    def forward(self, input_seq:torch.Tensor, input_label:torch.Tensor) -> tuple: # (torch.Tensor, torch.Tensor, torch.Tensor)
        # input_seq: (batch_size, max_seq_len)
        # input_label: (batch_size)
        if self.denosing_rate > 0:
            denosied_input_seq = input_seq[:, 1:-1].clone() # (batch_size, max_seq_len - 2) # <bos> and <eos> are not noised
            denosing_mask = torch.rand_like(denosied_input_seq, dtype=torch.float32) < self.denosing_rate
            denosied_input_seq[denosing_mask] = self.unk_id

            input_seq = torch.cat([input_seq[:, :1], denosied_input_seq, input_seq[:, -1:]], dim=-1) # (batch_size, max_seq_len)

        label_embed = self.label_embed(input_label) # (batch_size, embed_size)

        if self.model_type in ['rnn', 'gru']:
            # Encoder
            input_embed_encoder = self.seq_embed(input_seq) # (batch_size, max_seq_len, embed_size)
            input_length = torch.sum(input_seq != self.pad_id, dim=-1) # (batch_size)
            sorted_length, sorted_idx = torch.sort(input_length, descending=True)
            sorted_encoder_input = input_embed_encoder[sorted_idx] # (batch_size, max_seq_len, embed_size)
            packed_encoder_input = pack_padded_sequence(sorted_encoder_input, sorted_length.data.tolist(), batch_first=True)

            _, encoder_hidden = self.encoder(packed_encoder_input) # (batch_size, max_seq_len, output_size), (num_layers, batch_size, hidden_size)

            # Autoencoder process
            if self.variational_type in ['VAE', 'CVAE']: # Variational Autoencoder
                enc_hidden = encoder_hidden.clone() # (num_layers, batch_size, hidden_size)
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

                # Define decoder input
                input_embed_decoder = self.seq_embed(input_seq[:, :-1]) # (batch_size, max_seq_len-1, embed_size) # Remove <eos> token
                if self.variational_type == 'CVAE':
                    input_embed_decoder = input_embed_decoder + label_embed.unsqueeze(1).repeat(1, input_embed_decoder.size(1), 1) # (batch_size, max_seq_len-1, embed_size)
                input_length = input_length - 1 # (batch_size) # Remove <eos> token for decoder input
                sorted_length, sorted_idx = torch.sort(input_length, descending=True)
                sorted_decoder_input = input_embed_decoder[sorted_idx] # (batch_size, max_seq_len-1, embed_size)
                packed_decoder_input = pack_padded_sequence(sorted_decoder_input, sorted_length.data.tolist(), batch_first=True)
            else: # Deterministic Autoencoder
                mu = None
                logvar = None

                # Define decoder input
                input_embed_decoder = self.seq_embed(input_seq[:, :-1]) # (batch_size, max_seq_len-1, embed_size) # Remove <eos> token
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
            input_embed_encoder = self.seq_embed(input_seq) # (batch_size, max_seq_len, embed_size)
            encoder_pos_embed = self.pos_embed(encoder_pos_ids) # (max_seq_len, embed_size)

            encoder_input = input_embed_encoder + encoder_pos_embed
            src_key_padding_mask = (input_seq == self.pad_id) # (batch_size, max_seq_len)
            encoder_output = self.encoder(encoder_input, src_key_padding_mask=src_key_padding_mask) # (batch_size, max_seq_len, embed_size)

            # Autoencoder process
            if self.variational_type in ['VAE', 'CVAE']: # Variational Autoencoder
                enc_hidden = encoder_output.mean(dim=1) # (batch_size, embed_size)
                mu = self.variational_mu(enc_hidden) # (batch_size, latent_size)
                logvar = self.variational_logvar(enc_hidden) # (batch_size, latent_size)
                # Reparameterization Trick
                std = torch.exp(0.5 * logvar) # (batch_size, latent_size)
                eps = torch.randn_like(std) # (batch_size, latent_size)
                noise = mu + eps * std # (batch_size, latent_size)
                noise_embed = self.variational_hidden(noise) # (batch_size, embed_size)
                noise_embed = noise_embed.unsqueeze(1).repeat(1, self.max_seq_len-1, 1) # (batch_size, max_seq_len-1, embed_size)

                # Define decoder input
                decoder_pos_ids = torch.arange(0, input_seq[:, :-1].size(1), dtype=torch.long, device=input_seq.device) # (max_seq_len-1)
                input_embed_decoder = self.seq_embed(input_seq[:, :-1]) # (batch_size, max_seq_len-1, embed_size) # Remove <eos> token
                decoder_pos_embed = self.pos_embed(decoder_pos_ids) # (max_seq_len-1, embed_size)
                if self.variational_type == 'CVAE':
                    input_embed_decoder = input_embed_decoder + decoder_pos_embed + noise_embed + label_embed.unsqueeze(1).repeat(1, input_embed_decoder.size(1), 1)
                else:
                    input_embed_decoder = input_embed_decoder + decoder_pos_embed + noise_embed # (batch_size, max_seq_len-1, embed_size)
            else: # Deterministic Autoencoder
                mu = None
                logvar = None

                decoder_pos_ids = torch.arange(0, input_seq[:, :-1].size(1), dtype=torch.long, device=input_seq.device) # (max_seq_len-1)
                input_embed_decoder = self.seq_embed(input_seq[:, :-1]) # (batch_size, max_seq_len-1, embed_size)

                decoder_pos_embed = self.pos_embed(decoder_pos_ids) # (max_seq_len-1, embed_size)
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

    def inference(self, input_seq:torch.Tensor, input_label:torch.Tensor) -> tuple:
        # input_seq: (batch_size, max_seq_len)
        # input_label: (batch_size)
        if self.denosing_rate > 0:
            denosied_input_seq = input_seq[:, 1:-1].clone() # (batch_size, max_seq_len - 2) # <bos> and <eos> are not noised
            denosing_mask = torch.rand_like(denosied_input_seq, dtype=torch.float32) < self.denosing_rate
            denosied_input_seq[denosing_mask] = self.unk_id

            input_seq = torch.cat([input_seq[:, :1], denosied_input_seq, input_seq[:, -1:]], dim=-1) # (batch_size, max_seq_len)

        label_embed = self.label_embed(input_label) # (batch_size, embed_size)
        generated_sequence = torch.tensor([self.bos_id] * input_seq.size(0), dtype=torch.long, device=input_seq.device).unsqueeze(1) # (batch_size, 1)
        output_prob = torch.zeros(input_seq.size(0), 1, self.vocab_size, dtype=torch.float32, device=input_seq.device) # (batch_size, 1, vocab_size)

        if self.model_type in ['rnn', 'gru']:
            # Encoder
            input_embed_encoder = self.seq_embed(input_seq) # (batch_size, max_seq_len, embed_size)
            input_length = torch.sum(input_seq != self.pad_id, dim=-1) # (batch_size)
            sorted_length, sorted_idx = torch.sort(input_length, descending=True)
            sorted_encoder_input = input_embed_encoder[sorted_idx] # (batch_size, max_seq_len, embed_size)
            packed_encoder_input = pack_padded_sequence(sorted_encoder_input, sorted_length.data.tolist(), batch_first=True)

            _, encoder_hidden = self.encoder(packed_encoder_input) # (batch_size, max_seq_len, output_size), (num_layers, batch_size, hidden_size)

            # Autoencoder Process
            if self.variational_type in ['VAE', 'CVAE']: # Variational Autoencoder
                enc_hidden = encoder_hidden.clone() # (num_layers, batch_size, hidden_size)
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

            # Decoder
            for step in range(self.max_seq_len-1): # max_seq_len-1: <bos> is already generated
                input_embed_decoder = self.seq_embed(generated_sequence) # (batch_size, seq_len, embed_size)
                if self.variational_type == 'CVAE':
                    input_embed_decoder = input_embed_decoder + label_embed.repeat(1, input_embed_decoder.size(1), 1) # (batch_size, seq_len, embed_size)
                decoder_input_length = torch.sum(generated_sequence != self.pad_id, dim=-1) # (batch_size)
                sorted_length, sorted_idx = torch.sort(decoder_input_length, descending=True)
                sorted_decoder_input = input_embed_decoder[sorted_idx] # (batch_size, max_seq_len-1, embed_size)
                packed_decoder_input = pack_padded_sequence(sorted_decoder_input, sorted_length.data.tolist(), batch_first=True)

                decoder_output, _ = self.decoder(packed_decoder_input, encoder_hidden) # (batch_size, seq_len, output_size), (num_layers, batch_size, hidden_size)
                decoder_output, _ = pad_packed_sequence(decoder_output, batch_first=True) # (batch_size, seq_len, output_size)
                _, reversed_idx = torch.sort(sorted_idx) # (batch_size)
                decoder_output = decoder_output[reversed_idx] # (batch_size, max_seq_len-1, output_size)

                next_word_pred = self.output(decoder_output[:, -1, :]) # (test_batch_size, vocab_size)
                next_word_pred = F.log_softmax(next_word_pred, dim=-1) # (test_batch_size, vocab_size)
                next_word = next_word_pred.argmax(dim=-1, keepdim=True) # (test_batch_size, 1)

                generated_sequence = torch.cat([generated_sequence, next_word], dim=-1) # (test_batch_size, seq_len+1)
                output_prob = torch.cat([output_prob, next_word_pred.unsqueeze(1)], dim=1) # (test_batch_size, seq_len+1, vocab_size)

            generated_sequence = generated_sequence[:, 1:] # (test_batch_size, max_seq_len-1) # <bos> is not included
            output_prob = output_prob[:, 1:] # (test_batch_size, max_seq_len-1, vocab_size) # <bos> is not included
            output_prob = output_prob.view(-1, output_prob.size(-1)) # (test_batch_size * (max_seq_len-1), vocab_size)

        else: # Transformer
            # Encoder
            encoder_pos_ids = torch.arange(0, input_seq.size(1), dtype=torch.long, device=input_seq.device) # (max_seq_len)
            input_embed_encoder = self.seq_embed(input_seq) # (test_batch_size, max_seq_len, embed_size)
            encoder_pos_embed = self.pos_embed(encoder_pos_ids) # (max_seq_len, embed_size)

            encoder_input = input_embed_encoder + encoder_pos_embed
            encoder_output = self.encoder(encoder_input) # (test_batch_size, max_seq_len, embed_size)

            # Autoencoder process
            if self.variational_type in ['VAE', 'CVAE']:
                enc_hidden = encoder_output.mean(dim=1) # (test_batch_size, embed_size)
                mu = self.variational_mu(enc_hidden) # (test_batch_size, latent_size)
                logvar = self.variational_logvar(enc_hidden) # (test_batch_size, latent_size)
                # Reparameterization Trick
                std = torch.exp(0.5 * logvar) # (test_batch_size, latent_size)
                eps = torch.randn_like(std) # (test_batch_size, latent_size)
                noise = mu + eps * std # (test_batch_size, latent_size)
                noise_embed = self.variational_hidden(noise) # (test_batch_size, embed_size)
            else: # Deterministic Autoencoder
                mu = None
                logvar = None

            # Decoder process
            for step in range(self.max_seq_len-1): # max_seq_len-1: <bos> is already generated
                decoder_pos_ids = torch.arange(0, generated_sequence.size(1), dtype=torch.long, device=input_seq.device) # (seq_len)
                input_embed_decoder = self.seq_embed(generated_sequence) # (test_batch_size, seq_len, embed_size)
                decoder_pos_embed = self.pos_embed(decoder_pos_ids) # (seq_len, embed_size)

                if self.variational_type == 'AE': # Deterministic Autoencoder
                    input_embed_decoder = input_embed_decoder + decoder_pos_embed
                elif self.variational_type == 'VAE':
                    input_embed_decoder = input_embed_decoder + decoder_pos_embed + noise_embed
                elif self.variational_type == 'CVAE':
                    input_embed_decoder = input_embed_decoder + decoder_pos_embed + noise_embed + label_embed.unsqueeze(1).repeat(1, input_embed_decoder.size(1), 1)

                tgt_mask = self.generate_square_subsequent_mask(generated_sequence.size(1), device=input_seq.device) # (seq_len, seq_len)
                tgt_key_padding_mask = (generated_sequence == self.pad_id) # (test_batch_size, seq_len)

                decoder_output = self.decoder(input_embed_decoder, encoder_output,
                                              tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask) # (test_batch_size, seq_len, embed_size)

                next_word_pred = self.output(decoder_output[:, -1, :]) # (test_batch_size, vocab_size)
                next_word_pred = F.log_softmax(next_word_pred, dim=-1) # (test_batch_size, vocab_size)
                next_word = next_word_pred.argmax(dim=-1, keepdim=True) # (test_batch_size, 1)

                generated_sequence = torch.cat([generated_sequence, next_word], dim=-1) # (test_batch_size, seq_len+1)
                output_prob = torch.cat([output_prob, next_word_pred.unsqueeze(1)], dim=1) # (test_batch_size, seq_len+1, vocab_size)

            generated_sequence = generated_sequence[:, 1:] # (test_batch_size, max_seq_len-1) # <bos> is not included
            output_prob = output_prob[:, 1:] # (test_batch_size, max_seq_len-1, vocab_size) # <bos> is not included
            output_prob = output_prob.view(-1, output_prob.size(-1)) # (test_batch_size * (max_seq_len-1), vocab_size)

        return output_prob, generated_sequence, mu, logvar

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        mask = torch.tril(torch.ones(sz, sz, dtype=torch.float, device=device))
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask

def kl_annealing(epoch:int, max_epoch:int, start_val:float=0, end_val:float=1, annealing_rate:float=0.95) -> float:
    return end_val + (start_val-end_val) * annealing_rate ** epoch
