from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        if 'bert' in hug_name:
            return SegmentBERTEnocder(
                hug_name=hug_name,
                pad_id=pad_id,
                vocab_size=vocab_size,
                max_seg_len=None if kwargs['encoder_mask_type'] is None else kwargs['max_seg_len'],
                do_masked_lm=kwargs['do_masked_lm'],
            )
        else:
            return SegmentGPT2Enocder(
                hug_name=hug_name,
                pad_id=pad_id,
                vocab_size=vocab_size,
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


from transformers import BertModel, AutoModel, BertForMaskedLM
# from transformers.models.bert.modeling_bert import BertOnlyMLMHead

class SegmentBERTEnocder(nn.Module):
    def __init__(
        self,
        pad_id: int,
        hug_name: str = 'bert-base-chinese',
        vocab_size: int = 21131,
        max_seg_len: int = None,
        do_masked_lm: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id

        if do_masked_lm:
            mlm_model = BertForMaskedLM.from_pretrained(hug_name)
            mlm_model.resize_token_embeddings(vocab_size)
            self.encoder = mlm_model.bert
            self.cls = mlm_model.cls
        else:
            self.encoder = BertModel.from_pretrained(hug_name)
            self.encoder.resize_token_embeddings(vocab_size)

        self.embedding = self.encoder.get_input_embeddings()
        print(f'{self.embedding=}')
        print('-------')

        self.max_seg_len = max_seg_len
        self.vocab_size = vocab_size

    def forward(self, x: Tensor, **kwargs):
        embeds = self.embedding(x)
        # attn_mask = (x != self.pad_id).bool().to(x.device)
        attn_mask = self.generate_mask(x, self.max_seg_len)
        output = self.encoder(inputs_embeds=embeds, attention_mask=attn_mask)

        hidden_states = output.last_hidden_state

        return SegmentOutput(
            logits=None,
            embeds=embeds,
            decoder_hidden=hidden_states,
        )

    def generate_mask(self, x: Tensor, max_seg_len: int = None):
        attn_mask = (x != self.pad_id).bool().to(x.device)
        if max_seg_len:
            seq_len = x.size(1)
            # Make 3-dimension mask.
            seg_mask = (torch.ones((seq_len, seq_len))) == 1
            for i in range(seq_len):
                for j in range(1, min(max_seg_len + 1, seq_len - i)):
                    seg_mask[i, i + j] = False
            # `seg_mask` shape: (1, S, S)
            seg_mask = seg_mask[None, :, :].bool()
            attn_mask = attn_mask[:, None, :,] & seg_mask.to(x.device)
        return attn_mask

    def emb2vocab(self, hidden: Tensor):
        r"""Map embedding vectors to vocabulary."""
        return hidden @ self.embedding.weight.transpose(0, 1)

    def mlm_forward(self, x: Tensor, masked_labels: Tensor):
        attn_mask = self.generate_mask(x)
        output = self.encoder(input_ids=x, attention_mask=attn_mask)

        hidden_states = output.last_hidden_state

        # `logits` shape: (B, S, V)
        logits = F.softmax(self.cls(hidden_states), dim=-1)

        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), masked_labels.reshape(-1), ignore_index=0)

    @torch.no_grad()
    def perturbating_impact_matrix(self, x: Tensor, pertur_bz: int=256, upper_bound : int = 10):
        device = x.device
        input_ids, attention_mask = x, (x != 0)

        seq_len = input_ids.size(1) - 2
        ninput_ids = input_ids.unsqueeze(1).repeat(1, (2*seq_len -1), 1)
        nattention_mask = attention_mask.unsqueeze(1).repeat(1, (2*seq_len -1), 1)

        # Mask.
        for i in range(seq_len):
            if i > 0:
                ninput_ids[:, 2 * i - 1, i] = 103 # id of [mask]
                ninput_ids[:, 2 * i - 1, i + 1] = 103 # id of [mask]
            ninput_ids[:, 2 * i, i + 1] = 103 # id of [mask]

        batch_num = ninput_ids.size(0) * ninput_ids.size(1)
        if batch_num % pertur_bz == 0:
            batch_num = batch_num // pertur_bz
        else:
            batch_num = batch_num // pertur_bz + 1

        # `ninput_ids` shape: ( B*(2*S-1), S)
        ninput_ids = ninput_ids.view(-1, ninput_ids.size(-1))
        nattention_mask = nattention_mask.view(-1, nattention_mask.size(-1))
        small_batches = [{
            'input_ids': ninput_ids[num*pertur_bz : (num+1)*pertur_bz].to(device),
            'attention_mask': nattention_mask[num*pertur_bz : (num+1)*pertur_bz].to(device),
        } for num in range(batch_num)]

        # `vectors` shape: (B*(2*S-1), S, H)
        vectors = None
        for input in small_batches:
            if vectors is None:
                vectors = self.encoder(**input).last_hidden_state.detach()
                continue
            vectors = torch.cat([vectors, self.encoder(**input).last_hidden_state.detach()], dim=0)

        # `vec` shape (B, (2*S-1), S, H)
        new_size = (input_ids.size(0), -1, vectors.size(1), vectors.size(2))
        vec = vectors.view(new_size)

        all_dis = []
        for i in range(1, seq_len): # decide whether the i-th character and the (i+1)-th character should be in one word
            d1 = self.dist(vec[:, 2 * i, i + 1], vec[:, 2 * i - 1, i + 1])
            d2 = self.dist(vec[:, 2 * i - 2, i], vec[:, 2 * i - 1, i])
            d = (d1 + d2) / 2
            all_dis.append(d)

        # `all_dis` shape: (B, S-3)
        all_dis = torch.stack(all_dis, dim=1)

        # if d > upper_bound, then we combine the two tokens, else if d <= lower_bound then we segment them.
        labels = torch.where(all_dis>=upper_bound, 1, 0)
        # labels = torch.where(all_dis>=upper_bound, 1, 0)

        # # (B, S)
        # labels = torch.cat([
        #     torch.zeros(labels.size(0), 2).to(device),
        #     labels,
        #     torch.zeros(labels.size(0), 1).to(device)
        # ], dim=-1)

        # shape (S-3, B)
        # wo [CLS] [SEP] and relation matrix between each token.
        return labels.transpose(0, 1)


    @torch.no_grad()
    def dist(self, x, y):
        return torch.sqrt(((x - y)**2).sum(-1))




class SegmentGPT2Enocder(nn.Module):
    def __init__(
        self,
        pad_id: int,
        hug_name: str = 'ckiplab/gpt2-base-chinese',
        vocab_size: int = 21131,
        **kwargs,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id

        self.encoder = AutoModel.from_pretrained(hug_name)
        self.encoder.resize_token_embeddings(vocab_size)

        self.embedding = self.encoder.get_input_embeddings()
        print(f'{self.embedding=}')
        print('-------')


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

    def emb2vocab(self, hidden: Tensor):
        r"""Map embedding vectors to vocabulary."""
        return hidden @ self.embedding.weight.transpose(0, 1)
