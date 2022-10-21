# standara Library Modules
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ClassificationModel(nn.Module):
    def __init__(self, args:argparse.Namespace) -> None:
        super(ClassificationModel, self).__init__()
        self.model_type = args.model_type
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.dropout_rate = args.dropout_rate
        self.vocab_size = args.vocab_size
        self.bos_id = args.bos_id
        self.pad_id = args.pad_id
        self.unk_id = args.unk_id
        self.max_seq_len = args.max_seq_len
        self.activation_func = args.activation_func
        self.num_classes = args.num_classes

        # Embedding_layer
        self.seq_embed = nn.Sequential(
            nn.Embedding(self.vocab_size, self.embed_size, padding_idx=self.pad_id),
            nn.Dropout(self.dropout_rate)
        )
        self.pos_embed = nn.Sequential(
            nn.Embedding(self.max_seq_len, self.embed_size),
            nn.Dropout(self.dropout_rate)
        )

        # Activation layer
        if self.activation_func == 'relu':
            self.activation = nn.ReLU()
        elif self.activation_func == 'tanh':
            self.activation = nn.Tanh()
        elif self.activation_func == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif self.activation_func == 'gelu':
            self.activation = nn.GELU()

        # Encoder layer
        if self.model_type == 'rnn':
            self.encoder = nn.RNN(input_size=self.embed_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=False,
                                  dropout=self.dropout_rate,
                                  batch_first=True)

            self.classifier = nn.Sequential(
                nn.Linear(self.num_layers * self.hidden_size, self.hidden_size // 2),
                self.activation,
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_size // 2, self.num_classes)
            )
        elif self.model_type == 'gru':
            self.encoder = nn.GRU(input_size=self.embed_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=False,
                                  dropout=self.dropout_rate,
                                  batch_first=True)

            self.classifier = nn.Sequential(
                nn.Linear(self.num_layers * self.hidden_size, self.hidden_size // 2),
                self.activation,
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_size // 2, self.num_classes)
            )
        elif self.model_type == 'lstm':
            self.encoder = nn.LSTM(input_size=self.embed_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=False,
                                  dropout=self.dropout_rate,
                                  batch_first=True)

            self.classifier = nn.Sequential(
                nn.Linear(self.num_layers * self.hidden_size, self.hidden_size // 2),
                self.activation,
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_size // 2, self.num_classes)
            )
        elif self.model_type == 'cnn':
            each_out_size = self.hidden_size // 3
            self.encoder_k3 = nn.Conv1d(in_channels=self.embed_size, out_channels=each_out_size, kernel_size=3, padding='same')
            self.encoder_k4 = nn.Conv1d(in_channels=self.embed_size, out_channels=each_out_size, kernel_size=4, padding='same')
            self.encoder_k5 = nn.Conv1d(in_channels=self.embed_size, out_channels=each_out_size, kernel_size=5, padding='same')

            self.classifier = nn.Sequential(
                nn.Linear(each_out_size * 3, self.hidden_size // 2),
                self.activation,
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_size // 2, self.num_classes)
            )
        elif self.model_type == 'transformer':
            enc_layer = nn.TransformerEncoderLayer(d_model=self.embed_size, nhead=8, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer=enc_layer, num_layers=6)

            self.classifier = nn.Sequential(
                nn.Linear(self.embed_size, self.embed_size // 2),
                self.activation,
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.embed_size // 2, self.num_classes)
            )

    def forward(self, input_seq:torch.Tensor) -> torch.Tensor:
        # input_seq: [batch_size, seq_len]

        if self.model_type in ['rnn', 'gru', 'lstm']:
            # Embedding
            input_embed = self.seq_embed(input_seq)
            input_length = torch.sum(input_seq != self.pad_id, dim=-1).cpu() # (batch_size)
            sorted_length, sorted_idx = torch.sort(input_length, descending=True)
            sorted_encoder_input = input_embed[sorted_idx] # (batch_size, max_seq_len, embed_size)
            packed_encoder_input = pack_padded_sequence(sorted_encoder_input, sorted_length.data.tolist(), batch_first=True)

            # Encoder
            _, encoder_hidden = self.encoder(packed_encoder_input) # (num_layers, batch_size, hidden_size)
            if self.model_type == 'lstm':
                encoder_hidden = encoder_hidden[0] # (num_layers, batch_size, hidden_size) # Use only h_n, not c_n
            encoder_hidden = encoder_hidden.view(-1, self.num_layers * self.hidden_size) # (batch_size, num_layers * hidden_size)

            # Classifier
            logits = self.classifier(encoder_hidden) # (batch_size, num_classes)
        elif self.model_type == 'cnn':
            # Embedding
            input_embed = self.seq_embed(input_seq) # (batch_size, max_seq_len, embed_size)
            input_embed = input_embed.permute(0, 2, 1) # (batch_size, embed_size, max_seq_len)

            # Encoder
            enc1_output = torch.max(self.encoder_k3(input_embed), dim=-1)[0] # (batch_size, hidden_size // 3)
            enc2_output = torch.max(self.encoder_k4(input_embed), dim=-1)[0] # (batch_size, hidden_size // 3)
            enc3_output = torch.max(self.encoder_k5(input_embed), dim=-1)[0] # (batch_size, hidden_size // 3)

            enc_output = torch.cat([enc1_output, enc2_output, enc3_output], dim=1) # (batch_size, hidden_size)

            # Classifier
            logits = self.classifier(enc_output) # (batch_size, num_classes)
        elif self.model_type == 'transformer':
            # Embedding
            input_embed = self.seq_embed(input_seq)
            pos_ids = torch.arange(0, input_seq.size(1), dtype=torch.long, device=input_seq.device) # (max_seq_len)
            pos_embed = self.pos_embed(pos_ids) # (max_seq_len, embed_size)

            encoder_input = input_embed + pos_embed # (batch_size, max_seq_len, embed_size)
            src_key_padding_mask = (input_seq == self.pad_id) # (batch_size, max_seq_len)

            # Encoder
            encoder_output = self.encoder(encoder_input, src_key_padding_mask=src_key_padding_mask) # (batch_size, max_seq_len, embed_size)
            encoder_output = encoder_output.mean(dim=1) # (batch_size, embed_size)

            # Classifier
            logits = self.classifier(encoder_output) # (batch_size, num_classes)

        return logits
