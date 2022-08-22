import torch
import torch.nn as nn
from torch import Tensor

from codes._util import init_module


def SegmentDecoder(
    d_model: int,
    dec_n_layers: int,
    dec_n_heads: int = None,
    dropout: float = 0.1,
) -> None:

    if dec_n_heads:
        model = SegmentDecoderTransformer(
            d_model=d_model,
            dec_n_layers=dec_n_layers,
            dec_n_heads=dec_n_heads,
            dropout=dropout,
        )
    else:
        model = SegmentDecoderLSTM(
            d_model=d_model,
            dec_n_layers=dec_n_layers,
            dropout=dropout,
        )

    return model.apply(init_module)

class SegmentDecoderLSTM(nn.Module):
    def __init__(
        self,
        d_model: int,
        dec_n_layers: int,
        dropout: float = 0.1,
        is_narrowed: bool = False,
    ) -> None:
        super().__init__()

        self.dec_n_layers = dec_n_layers

        dim_narrow = d_model // 4 if is_narrowed else d_model
        print(is_narrowed)
        print(dim_narrow)
        if is_narrowed:
            self.narrow_linear = nn.Linear(d_model, dim_narrow)
            self.narrow_segment_decoder_ff = nn.Linear((1+dec_n_layers) * d_model, (1+dec_n_layers) * dim_narrow)
            self.recover_linear = nn.Linear(dim_narrow, d_model)

        self.segment_lstm = nn.LSTM(
            input_size=dim_narrow,
            hidden_size=dim_narrow,
            num_layers=dec_n_layers,
            dropout=dropout
        )

        self.segment_decoder_ff = nn.Linear(d_model, (1+dec_n_layers) * d_model)

        self.droupout = nn.Dropout(p=dropout)

    def gen_start_symbol_hidden(self, encoder_output: Tensor):
        return self.segment_decoder_ff(encoder_output)
        # return torch.tanh(self.segment_decoder_ff(encoder_output))

    def forward(self, seg_start_hidden: Tensor, seg_embeds: Tensor) -> Tensor:
        """Estimating the probabilities of each char in segment.

        Parameters
        ----------
        seg_start_hidden : Tensor
            shape: (1, B, (1 + dec_n_layers)*d_model)
        seg_embeds : Tensor
            shape: (max_seg_len, B, d_model)

        Returns
        -------
        Tensor
            shape: (1+max_seg_len, B, d_model)
        """
        self.segment_lstm.flatten_parameters()

        if hasattr(self, 'narrow_linear'):
            # `seg_embeds` shape: (max_seg_len, B, dim_narrow)
            seg_start_hidden = self.narrow_segment_decoder_ff(seg_start_hidden)
            seg_embeds = self.narrow_linear(seg_embeds)

        d_model = seg_embeds.size(-1)
        batch_size = seg_embeds.size(1)

        # `start_symbol` sahpe: (1, B, d_model)
        start_symbol = seg_start_hidden[:, :, :d_model]

        # `decoder_h_init` shape: (B, dec_n_layers * d_model)
        # `decoder_h_init` shape: (B, dec_n_layers, d_model)
        # `decoder_h_init` shape: (dec_n_layers, B, d_model)
        decoder_h_init = seg_start_hidden[:, :, d_model:].squeeze(0)
        decoder_h_init = decoder_h_init.view(batch_size, self.dec_n_layers, -1)
        decoder_h_init = torch.tanh(decoder_h_init.transpose(0, 1).contiguous())

        # `decoder_c_init` shape: (dec_n_layers, B, d_model)
        decoder_c_init = torch.zeros_like(decoder_h_init)

        # `decoder_input` shape: (1+max_seg_len, B, d_model)
        decoder_input = torch.cat([start_symbol, seg_embeds], dim=0).contiguous()

        # `decoder_output` shape: (max_seg_len+1, B, d_model)
        decoder_output, _ = self.segment_lstm(decoder_input, (decoder_h_init, decoder_c_init))

        if hasattr(self, 'narrow_linear'):
            decoder_output = self.recover_linear(decoder_output)

        # shape: (1+max_seg_len, B, d_model)
        return self.droupout(decoder_output)


class SegmentDecoderTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        dec_n_layers: int,
        dec_n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=d_model, nhead=dec_n_heads)
        self.segment_model = nn.TransformerEncoder(encoder_layer, num_layers=dec_n_layers)

        self.dropout = nn.Dropout(p=dropout)
        self.segment_decoder_ff = nn.Linear(d_model, d_model)

    def gen_start_symbol_hidden(self, encoder_output: Tensor):
        return torch.tanh(self.segment_decoder_ff(encoder_output))

    def forward(self, seg_start_hidden: Tensor, seg_embeds: Tensor) -> Tensor:
        """Estimating the probabilities of each char in segment.

        Parameters
        ----------
        seg_start_hidden : Tensor
            shape: (1, B, d_model)
        seg_embeds : Tensor
            shape: (max_seg_len, B, d_model)

        Returns
        -------
        Tensor
            shape: (1+max_seg_len, B, d_model)
        """

        # `decoder_input` shape: (1+max_seg_len, B, d_model)
        decoder_input = torch.cat([seg_start_hidden, seg_embeds], dim=0).contiguous()

        # ``True`` position will be ignored.
        # `mask` shape: (1+max_seg_len, 1+max_seg_len)
        mask = torch.ones(decoder_input.size(0), decoder_input.size(0)).triu(1).bool()

        # `decoder_output` shape: (1+max_seg_len, B, d_model)
        decoder_output = self.segment_model(src=decoder_input, mask=mask.to(seg_embeds.device))

        # Return shape: (1+max_seg_len, B, d_model)
        return self.dropout(decoder_output)

# import torch
# def _generate_square_subsequent_mask(sz):
#     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#     return mask

# m = _generate_square_subsequent_mask(10)
# m.size()
# m


# def subsequent_mask(x, pad_id):
#     r"""Mask out subsequent positions."""
#     # `lm_mask` shape: (1, S, S)
#     lm_mask = torch.ones(1, x.size(1), x.size(1)).triu(1).bool()
#     # `mask` shape: (B, S, 1)
#     mask = (x == pad_id).unsqueeze(2)
#     # `mask | lm_mask` shape: (B, S, S)
#     return mask.to(x.device) | lm_mask.to(x.device)

# x = torch.randint(0, 10, (3, 5))
# subsequent_mask(x, 2)