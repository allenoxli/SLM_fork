from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from codes._model_output import SegmentOutput
from codes._transformer_module import (subsequent_mask, PositionalEncodingLearned, PositionalEncoding, Decoder)
from codes._util import init_module

class SegEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
        vocab_size: int,
        init_embedding: Dict,
        is_pos: bool,
        max_len: int,
        pad_id: int,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=d_model, padding_idx=pad_id)
        init_module(self.embedding)

        if is_pos:
            print(f'{is_pos=}')
            self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
            # self.positional_encoding.apply(init_module)

        if init_embedding:
            embed = init_embedding['embedding']
            pos_embed = init_embedding['position']
            print(f'{embed.shape=}')
            assert embed.shape[0] == vocab_size
            assert embed.shape[1] == d_model
            self.embedding = nn.Embedding.from_pretrained(nn.Parameter(torch.from_numpy(embed).float()))
            if pos_embed is not None:
                print(f'{pos_embed.shape=}')
                self.positional_encoding.pe = nn.Embedding.from_pretrained(nn.Parameter(torch.from_numpy(pos_embed[:max_len, :]).float()))

        self.embedding2vocab = nn.Linear(d_model, vocab_size)
        self.embedding2vocab.weight = self.embedding.weight

    def emb2vocab(self, hidden: Tensor):
        r"""Map embedding vectors to vocabulary."""
        # return hidden @ self.embedding.weight.transpose(0, 1)
        return self.embedding2vocab(hidden)

    def forward(self, x: Tensor):
        # `embeds` shape: (B, S, d_model)
        embeds = self.embedding(x)
        if hasattr(self, 'positional_encoding'):
            embeds = self.positional_encoding(embeds)

        return embeds


def SegmentEncoder(
        d_model: int,
        d_ff: int,
        dropout: float,
        embedding: SegEmbedding,
        max_len: int,
        n_layers: int,
        n_heads: int,
        vocab_size: int,
        pad_id: int,
        **kwargs,
):

    if n_heads:
        return SegmentTransformerEnocder(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
            n_layers=n_layers,
            n_heads=n_heads,
            vocab_size=vocab_size,
            pad_id=pad_id,
            embedding=embedding,
        )

    return SegmentLSTMEnocder(
        d_model=d_model,
        dropout=dropout,
        n_layers=n_layers,
        embedding=embedding,
    )

class SegmentTransformerEnocder(nn.Module):
    def __init__(
        self,
        embedding: SegEmbedding,
        d_model: int,
        d_ff: int,
        dropout: float,
        n_layers: int,
        n_heads: int,
        pad_id: int,
        **kwargs
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = embedding

        # Transformer-based encoder.
        self.encoder = Decoder(d_model=d_model, d_ff=d_ff, n_layers=n_layers, n_heads=n_heads, dropout=dropout)

        # encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=d_ff , nhead=n_heads, batch_first=True)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.encoder.apply(init_module)

    def forward(self, x: Tensor, **kwargs):

        # `embeds` shape: (B, S, d_model)
        embeds = self.embedding(x)

        lm_mask = subsequent_mask(x, self.pad_id)
        # lm_mask = torch.ones(x.size(1), x.size(1)).triu(1)
        # src_key_padding_mask = (x == self.pad_id).bool()

        # `hidden_states` shape: (B, S, d_model)
        # hidden_states = self.encoder(embeds, mask=lm_mask.to(x.device), src_key_padding_mask=src_key_padding_mask)
        hidden_states = self.encoder(embeds, mask=lm_mask.to(x.device))

        logits = self.embedding.emb2vocab(hidden_states)

        return SegmentOutput(
            logits=logits[:, :-1, :],
            embeds=embeds,
            decoder_hidden=hidden_states,
        )

    def get_weight(self):
        return self.encoder.decoder_layers[-1].feedforward.layers[2].weight[0]

class SegmentLSTMEnocder(nn.Module):
    def __init__(
        self,
        embedding: SegEmbedding,
        d_model: int,
        dropout: float,
        n_layers: int,
        **kwargs
    ):
        super().__init__()

        self.embedding = embedding

        # Transformer-based encoder.
        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
        )

        # `h_init_state` shpae: (n_layers * num_directions, B, d_model)
        # `c_init_state` shpae: (n_layers * num_directions, B, d_model)
        self.h_init_state = nn.Parameter(torch.zeros(n_layers, 1, d_model))
        self.c_init_state = nn.Parameter(torch.zeros(n_layers, 1, d_model))

        self.encoder.apply(init_module)

    def forward(self, x: Tensor, **kwargs):
        self.encoder.flatten_parameters()

        # `embeds` shape: (B, S, d_model)
        embeds = self.embedding(x)

        # Make LSTM init states (h, c).
        # `h` shape: (n_layers * num_directions, B, d_model)
        # `c` shape: (n_layers * num_directions, B, d_model)
        h = self.h_init_state.expand(-1, x.size(0), -1).contiguous()
        c = self.c_init_state.expand(-1, x.size(0), -1).contiguous()

        # `hidden_states` shape: (B, S, d_model)
        hidden_states, _ = self.encoder(embeds, (h, c))

        logits = self.embedding.emb2vocab(hidden_states)

        return SegmentOutput(
            logits=logits[:, :-1, :],
            embeds=embeds,
            decoder_hidden=hidden_states,
        )

    def get_weight(self):
        return self.encoder.all_weights[-1][1][0]


# encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
# transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
# src = torch.rand(10, 32, 256)
# mask = torch.ones(32, 32).triu(1)
# src_key_padding_mask = torch.ones(10, 32)

# out = transformer_encoder(src, mask=mask, src_key_padding_mask=src_key_padding_mask)