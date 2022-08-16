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
        bidirectional: bool= False,
        hug_name: str = None,
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
            **kwargs,
        )
    if bidirectional:
        return SegmentBiLSTMEnocder(
            d_model=d_model,
            dropout=dropout,
            n_layers=n_layers,
            embedding=embedding,
            **kwargs,
        )

    if hug_name:
        return SegmentBERTEnocder(
            hug_name=hug_name,
            pad_id=pad_id,
            vocab_size=vocab_size,
            max_seg_len=None if kwargs['encoder_mask_type'] is None else kwargs['max_seg_len']
        )

    return SegmentLSTMEnocder(
        d_model=d_model,
        dropout=dropout,
        n_layers=n_layers,
        embedding=embedding,
        **kwargs,
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

        self.encoder_input_dropout = nn.Dropout(kwargs['encoder_input_dropout_rate'])

        self.encoder.apply(init_module)

    def forward(self, x: Tensor, **kwargs):
        self.encoder.flatten_parameters()

        # `embeds` shape: (B, S, d_model)
        embeds = self.encoder_input_dropout(self.embedding(x))

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

class SegmentBiLSTMEnocder(nn.Module):
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
            bidirectional=True,
        )

        # `h_init_state` shpae: (n_layers * num_directions, B, d_model)
        # `c_init_state` shpae: (n_layers * num_directions, B, d_model)
        self.h_init_state = nn.Parameter(torch.zeros(n_layers*2, 1, d_model))
        self.c_init_state = nn.Parameter(torch.zeros(n_layers*2, 1, d_model))

        self.encoder_input_dropout = nn.Dropout(kwargs['encoder_input_dropout_rate'])

        self.encoder.apply(init_module)

    def forward(self, x: Tensor, **kwargs):
        self.encoder.flatten_parameters()

        # `embeds` shape: (B, S, d_model)
        embeds = self.encoder_input_dropout(self.embedding(x))

        # Make LSTM init states (h, c).
        # `h` shape: (n_layers * num_directions, B, d_model)
        # `c` shape: (n_layers * num_directions, B, d_model)
        # h = torch.cat([self.h_init_state]*x.size(0), dim=1)
        h = self.h_init_state.expand(-1, x.size(0), -1).contiguous()
        c = self.c_init_state.expand(-1, x.size(0), -1).contiguous()

        # `hidden_states` shape: (B, S, num_directions*d_model)
        hidden_states, _ = self.encoder(embeds, (h, c))


        # `hidden_states` shape: (B, S, num_directions, d_model)
        # `hidden_states` shape: (B, S, d_model)
        hidden_states = hidden_states.reshape(embeds.size(0), embeds.size(1), 2, -1)
        hidden_states = (hidden_states[:, :, 0, :] + hidden_states[:, :, 1, :].flip(1)) / 2
        # hidden_states = hidden_states[:, :, 0, :]


        logits = self.embedding.emb2vocab(hidden_states)

        return SegmentOutput(
            logits=logits[:, :-1, :],
            embeds=embeds,
            decoder_hidden=hidden_states,
        )

    def get_weight(self):
        return self.encoder.all_weights[-1][1][0]


from transformers import BertModel
class SegmentBERTEnocder(nn.Module):
    def __init__(
        self,
        pad_id: int,
        hug_name: str = 'bert-base-chinese',
        vocab_size: int = 21131,
        max_seg_len: int = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id

        self.encoder = BertModel.from_pretrained(hug_name)
        self.encoder.resize_token_embeddings(vocab_size)

        self.embedding = self.encoder.get_input_embeddings()
        print(f'{self.embedding=}')
        print('-------')
        print('-------')
        print('-------')

        self.max_seg_len = max_seg_len

    def forward(self, x: Tensor, **kwargs):
        embeds = self.embedding(x)
        attn_mask = (x != self.pad_id).bool().to(x.device)
        output = self.encoder(inputs_embeds=embeds, attention_mask=attn_mask)

        hidden_states = output.last_hidden_state

        return SegmentOutput(
            logits=None,
            embeds=embeds,
            decoder_hidden=hidden_states,
        )

    def generate_mask(self, x: Tensor, max_seg_len: int = None):
        attn_mask = (x != self.pad_id).bool()
        if max_seg_len:
            # Make 3-dimension mask.
            seg_mask = self.generate_segment_mask(x.size(-1), max_seg_len)
            attn_mask = attn_mask[:, None, :] & seg_mask.to(x.device)
        return attn_mask

    def emb2vocab(self, hidden: Tensor):
        r"""Map embedding vectors to vocabulary."""
        return hidden @ self.embedding.weight.transpose(0, 1)

import os
# from codes.pytorch_hug_bert.modeling import BertModelSeg
# class SegmentBERTSegEnocder(nn.Module):
#     def __init__(
#         self,
#         pad_id: int,
#         hug_name: str = 'bert-base-chinese',
#         vocab_size: int = 21131,
#         **kwargs,
#     ) -> None:
#         super().__init__()
#         self.pad_id = pad_id

#         bert_path = 'bert-base-chinese/pytorch_model.bin'
#         print(os.path.exists(bert_path))
#         model_bert = BertModelSeg.from_pretrained('bert-base-chinese', state_dict=torch.load(bert_path))
#         model_bert.train()

#         self.encoder = model_bert
#         self.encoder.resize_token_embeddings(vocab_size)

#         self.embedding = self.encoder.get_input_embeddings()
#         print(f'{self.embedding=}')
#         print('-------')
#         print('-------')
#         print('-------')

#     def forward(self, x: Tensor, **kwargs):
#         attn_mask = (x != self.pad_id).bool().to(x.device)
#         output = self.encoder(input_ids=x, attention_mask=attn_mask)

#         hidden_states = output[0]

#         embeds = output[2]

#         return SegmentOutput(
#             logits=None,
#             embeds=embeds,
#             decoder_hidden=hidden_states,
#         )

#     def generate_mask(self, x: Tensor):
#         attn_mask = (x != self.pad_id).bool()
#         return attn_mask

#     def emb2vocab(self, hidden: Tensor):
#         r"""Map embedding vectors to vocabulary."""
#         return hidden @ self.encoder.embeddings.word_embeddings.weight.transpose(0, 1)


