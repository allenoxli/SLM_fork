from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from codes._util import init_module
from codes._model_output import SegmentOutput
from codes._segment_encoder import SegEmbedding, SegmentEncoder
from codes._segment_decoder import SegmentDecoder

class SegmentModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dec_n_layers: int,
        dec_n_heads: int,
        dec_dropout: float,
        dropout: float,
        max_len: int,
        n_layers: int,
        n_heads: int,
        vocab_size: int,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        punc_id: int = None,
        eng_id: int = None,
        num_id: int = None,
        init_embedding: Dict = None,
        **kwargs
    ):
        super().__init__()

        # Special token id.
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

        self.punc_id = punc_id
        self.eng_id = eng_id
        self.num_id = num_id

        self.embedding = SegEmbedding(
            d_model=d_model,
            dropout=dropout,
            vocab_size=vocab_size,
            init_embedding=init_embedding,
            is_pos=True if n_heads else False,
            max_len=max_len,
            pad_id=pad_id,
        )

        # Context Encoder.
        self.encoder = SegmentEncoder(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            embedding=self.embedding,
            max_len=max_len,
            n_layers=n_layers,
            n_heads=n_heads,
            vocab_size=vocab_size,
            pad_id=pad_id,
            init_embedding=init_embedding,
        )

        # Segment decoder.
        self.segment_decoder = SegmentDecoder(
            d_model=d_model,
            dec_n_layers=dec_n_layers,
            dec_n_heads=dec_n_heads,
            dropout=dec_dropout,
        )

    def forward(self, x: Tensor, **kwargs):
        return self.encoder(x)

    def segment_forward(
        self,
        x: Tensor,
        lm_output: SegmentOutput,
        max_seg_len: int,
        mode: str,
        lengths: torch.Tensor,
        loginf: float = 1000000.0,
        **kwargs,
    ):

        embeds = lm_output.embeds
        decoder_hidden = lm_output.decoder_hidden

        batch_size = x.size(0)
        max_len = max(lengths).item()

        # `neg_inf_vector` shape: (B)
        neg_inf_vector = torch.full_like(embeds[:, 0, 0], -loginf)

        # `x` shape (S, B)
        # `embeds` shape (S, B, d_model)
        # `encoder_output` shape (S, B, d_model)
        x = x.transpose(0, 1)
        embeds = embeds.transpose(0, 1)
        encoder_output = decoder_hidden.transpose(0, 1)

        # `logpy` shape: (max_len-1, max_seg_len, B)
        # log_probability
        logpy = neg_inf_vector.repeat(max_len - 1, max_seg_len, 1)
        logpy[0][0] = 0

        if self.punc_id is not None:
            # 找 這些 ID 位置。
            # `is_single` shape: (S, B)
            is_single = -loginf * ((x == self.punc_id) | (x == self.eng_id) | (x == self.num_id)).type_as(x)

        # Get start of segment hidden state.
        seg_dec_hiddens = self.segment_decoder.gen_start_symbol_hidden(encoder_output)

        for j_start in range(1, max_len - 1):
            j_end = j_start + min(max_seg_len, (max_len-1) - j_start)

            # Estimating each character probability of segment by giving
            # segment start hidden symbol and segment word embeddings.
            # `decoder_output` shape: (max_seg_len+1, B, H)
            decoder_output = self.segment_decoder(
                seg_start_hidden=seg_dec_hiddens[j_start-1, :, :].unsqueeze(0),
                seg_embeds=embeds[j_start:j_end, :, :],
            )

            # `decoder_logpy` shape: (max_seg_len+1, B, V)
            # `decoder_target` shape: (max_seg_len, B) , get the character's id.
            decoder_logpy = F.log_softmax(self.embedding.emb2vocab(decoder_output), dim=2)
            decoder_target = x[j_start:j_end, :]

            # 取出正確字對應的機率 form vocab size probabilities.
            # `target_logpy` shape: (max_seg_len, B)
            target_logpy = decoder_logpy[:-1, :, :].gather(dim=2, index=decoder_target.unsqueeze(-1)).squeeze(-1)

            # `tmp_logpy` shape: (B)
            tmp_logpy = torch.zeros_like(target_logpy[0])

            # `j` is a temporary `j_end`.
            for j in range(j_start, j_end):
                tmp_logpy = tmp_logpy + target_logpy[j - j_start, :]

                if j > j_start:
                    tmp_logpy = tmp_logpy + is_single[j, :]
                if j == j_start + 1:
                    tmp_logpy = tmp_logpy + is_single[j_start, :]
                logpy[j_start][j - j_start] = tmp_logpy + decoder_logpy[j - j_start + 1, :, self.eos_id]

        if mode == 'train':
            # Total_log_probability
            # `alpha` shape: (S-1, B)
            # log probability for generate [bos] at beginning is 0
            alpha = neg_inf_vector.repeat(max_len - 1, 1)
            alpha[0] = 0

            for j_end in range(1, max_len - 1):
                logprobs = []
                for j_start in range(max(0, j_end - max_seg_len), j_end):
                    logprobs.append(alpha[j_start] + logpy[j_start + 1][j_end - j_start - 1])

                alpha[j_end] = torch.logsumexp(torch.stack(logprobs), dim=0)

            # `index` shape: (1, B)
            # `total_length`: For normalizing the loss.
            index = (lengths - 2).unsqueeze(0)
            total_length = (lengths - 2).sum().item()

            NLL_loss = - torch.gather(input=alpha, dim=0, index=index)
            NLL_loss.clamp_(min=0.0)

            # assert NLL_loss.view(-1).size(0) == batch_size

            return NLL_loss.sum() / float(total_length)

        if mode == 'decode':

            ret = []
            # <BOS> is at the begin of sentence,
            # <PUNC> is at the end of sentece
            # sentence_length = true_length + 2
            for i in range(batch_size):
                alpha = [-loginf]*(lengths[i] - 1)
                prev = [-1]*(lengths[i] - 1)
                alpha[0] = 0.0
                for j_end in range(1, lengths[i] - 1):
                    for j_start in range(max(0, j_end - max_seg_len), j_end):
                        logprob = alpha[j_start] + logpy[j_start + 1][j_end - j_start - 1][i].item()
                        if logprob > alpha[j_end]:
                            alpha[j_end] = logprob
                            prev[j_end] = j_start

                j_end = lengths[i].item() - 2
                segment_lengths = []
                while j_end > 0:
                    prev_j = prev[j_end]
                    segment_lengths.append(j_end - prev_j)
                    j_end = prev_j

                segment_lengths = segment_lengths[::-1]
                ret.append(segment_lengths)

            return ret

    def generate_label(self, x: Tensor, lengths: Tensor, max_seg_len: int):
        r"""Generate the labels for classifier."""

        output = self(x)
        batch_segments = self.segment_forward(
            x=x,
            lm_output=output,
            max_seg_len=max_seg_len,
            mode='decode',
            lengths=lengths,
        )

        batch_labels = []
        for input_ids, segment in zip(x, batch_segments):
            label = torch.zeros(input_ids.size(0))
            # `label` including the labels of <bos> and <eos> token.
            # `-1+1` for correct segment index and BOS token.
            idx = [sum(segment[:i])-1+1 for i in range(1, len(segment)+1)]
            label[idx] = 1
            batch_labels.append(label)

        return batch_labels

    def log_param(self, writer, step):
        r"""Logging parameter value."""

        lm_param = self.encoder.get_weight()

        @torch.no_grad()
        def log_value(writer, step, vals, name):
            val = vals.cpu()
            q_pos = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
            q_val = torch.quantile(val, q_pos, dim=-1)
            writer.add_scalar(f'{name}/min', q_val[0].item(), step)
            writer.add_scalar(f'{name}/q1', q_val[1].item(), step)
            writer.add_scalar(f'{name}/q2', q_val[2].item(), step)
            writer.add_scalar(f'{name}/q3', q_val[3].item(), step)
            writer.add_scalar(f'{name}/max', q_val[4].item(), step)
            writer.add_scalar(f'{name}/mean', val.mean().item(), step)
            writer.add_scalar(f'{name}/std', val.std().item(), step)
            writer.add_scalar(f'{name}/sum', val.sum().item(), step)

        log_value(writer, step, lm_param, 'lm_param')
