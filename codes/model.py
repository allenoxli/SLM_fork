#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SLM Model
"""
# %%
import six
import copy
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


from codes._segment_encoder import SegEmbedding, SegmentEncoder
from codes._segment_decoder import SegmentDecoder
# %%

class SLMConfig(object):
    """Configuration for `SegmentalLM`."""

    def __init__(
        self,
        vocab_size,
        embedding_size=256,
        hidden_size=256,
        max_segment_length=4,
        encoder_layer_number=1,
        decoder_layer_number=1,
        encoder_input_dropout_rate=0.0,
        decoder_input_dropout_rate=0.0,
        encoder_dropout_rate=0.0,
        decoder_dropout_rate=0.0,
        punc_id=2,
        num_id=3,
        eos_id=5,
        eng_id=7
    ):
        """
        Constructs SLMConfig.
        """
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.max_segment_length = max_segment_length
        self.encoder_layer_number = encoder_layer_number
        self.decoder_layer_number = decoder_layer_number
        self.encoder_input_dropout_rate = encoder_input_dropout_rate
        self.decoder_input_dropout_rate = decoder_input_dropout_rate
        self.encoder_dropout_rate = encoder_dropout_rate
        self.decoder_dropout_rate = decoder_dropout_rate
        self.eos_id = eos_id
        self.punc_id = punc_id
        self.eng_id = eng_id
        self.num_id = num_id

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `SLMConfig` from a Python dictionary of parameters."""
        config = SLMConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `SLMConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_json_file(self, json_file):
        """Serializes this instance to a JSON file."""
        with open(json_file, "w") as writer:
            writer.write(self.to_json_string())

class SegmentalLM(nn.Module):
    def __init__(
        self,
        config,
        init_embedding = None,
        **kwargs,
    ):
        super(SegmentalLM, self).__init__()

        config = copy.deepcopy(config)

        self.config = config

        if init_embedding is not None:
            print('-----')
            assert np.shape(init_embedding)[0] == config.vocab_size
            assert np.shape(init_embedding)[1] == config.embedding_size
            shard_embedding = nn.Parameter(torch.from_numpy(init_embedding).float())
        else:
            shard_embedding = torch.zeros(config.vocab_size, config.embedding_size)
            nn.init.uniform_(shard_embedding, a=-1.0, b=1.0)

        n_heads = None
        dec_n_heads = None
        pad_id = 0
        bidirectional = False

        # self.embedding = nn.Embedding.from_pretrained(shard_embedding)
        self.embedding = SegEmbedding(
            d_model=config.embedding_size,
            dropout=0.1,
            vocab_size=config.vocab_size,
            init_embedding={
                'embedding': shard_embedding.detach().numpy(),
                'position': None,
            },
            is_pos=False if n_heads is None else True,
            max_len=32,
            pad_id=pad_id,
        )

        # self.embedding2vocab = nn.Linear(config.embedding_size, config.vocab_size)
        self.embedding2vocab = self.embedding.emb2vocab

        # Weight Tying
        # self.embedding2vocab.weight = self.embedding.weight

        # self.context_encoder = ContextEncoder(
        #     input_size=config.embedding_size,
        #     hidden_size=config.hidden_size,
        #     layer_number=config.encoder_layer_number,
        #     dropout_rate=config.encoder_dropout_rate
        # )

        self.context_encoder = SegmentEncoder(
            d_model=config.embedding_size,
            d_ff=config.hidden_size,
            dropout=config.encoder_dropout_rate,
            embedding=self.embedding,
            max_len=32,
            n_layers=config.encoder_layer_number,
            n_heads=n_heads,
            vocab_size=config.vocab_size,
            pad_id=pad_id,
            init_embedding={
                'embedding': shard_embedding.detach().numpy(),
                'position': None,
            },
            encoder_input_dropout_rate=config.encoder_input_dropout_rate,
            bidirectional=bidirectional,
            hug_name=kwargs['hug_name'],
            max_seg_len=config.max_segment_length,
            encoder_mask_type=kwargs['encoder_mask_type'],
            do_masked_lm=kwargs['do_masked_lm'],
        )

        if kwargs['hug_name'] is not None:
            self.embedding = self.context_encoder.embedding
            self.embedding2vocab = self.context_encoder.emb2vocab

        self.segment_decoder = SegmentDecoder(
            d_model=config.hidden_size,
            dec_n_layers=config.decoder_layer_number,
            dec_n_heads=dec_n_heads,
            dropout=config.decoder_dropout_rate,
            dim_narrow=kwargs['dim_narrow'],
        )

        # self.segment_decoder = SegmentDecoder2(
        #     hidden_size=config.hidden_size,
        #     output_size=config.embedding_size,
        #     layer_number=config.decoder_layer_number,
        #     dropout_rate=config.decoder_dropout_rate
        # )

        # self.decoder_h_transformation = nn.Linear(
        #     config.hidden_size,
        #     config.decoder_layer_number * config.hidden_size
        # )

        # self.encoder_input_dropout = nn.Dropout(p=config.encoder_input_dropout_rate)
        self.decoder_input_dropout = nn.Dropout(p=config.decoder_input_dropout_rate)

        # self.start_of_segment = nn.Linear(
        #     config.hidden_size,
        #     config.hidden_size
        # )

    def forward(
        self,
        x,
        lengths,
        segments=None,
        mode: str='unsupervised',
        no_single: bool=False,
        is_impacted: bool=False,
        upper_bound: int=10,
        **kwargs,
    ):
        if mode == 'supervised' and segments is None:
            raise ValueError('Supervised mode needs segmented text.')

        #input format: (seq_len, batch_size)
        x = x.transpose(0, 1).contiguous()

        #transformed format: (seq_len, batch_size)
        max_length = x.size(0)
        batch_size = x.size(1)

        loginf = 1000000.0

        max_length = max(lengths)

        lm_output = self.context_encoder(x.transpose(0, 1).contiguous())
        inputs = lm_output.embeds.transpose(0, 1)
        encoder_output = lm_output.decoder_hidden.transpose(0, 1)

        # `impact_matrix` shape: (S-3, B)
        impact_matrix = torch.zeros_like(x)
        if is_impacted:
            impact_matrix = self.context_encoder.perturbating_impact_matrix(x.transpose(0, 1).contiguous(), upper_bound=upper_bound)
            # `impact_matrix` shape: (S-2, B)
            impact_matrix = torch.cat([impact_matrix, torch.zeros(1, impact_matrix.size(-1)).to(impact_matrix.device)], dim=0)

        # `is_single` shape: (S, B)
        is_single = -loginf * ((x == self.config.punc_id) | (x == self.config.eng_id) | (x == self.config.num_id)).type_as(x)
        if mode == 'supervised' or no_single:
          is_single = torch.zeros_like(is_single)

        neg_inf_vector = torch.full_like(inputs[0,:,0], -loginf)

        # `logpy` shape: (S-1, max_seg_len, B)
        logpy = neg_inf_vector.repeat(max_length - 1, self.config.max_segment_length, 1)
        logpy[0][0] = 0

        # Make context encoder and segment decoder have different learning rate
        encoder_output = encoder_output * 0.5

        seg_dec_hiddens = self.segment_decoder.gen_start_symbol_hidden(encoder_output)

        # for j_start, j_len, j_end in schedule:
        for j_start in range(1, max_length - 1):
            j_end = j_start + min(self.config.max_segment_length, (max_length-1) - j_start)
            # Segment Decoder

            decoder_output = self.segment_decoder(
                seg_start_hidden=seg_dec_hiddens[j_start-1, :, :].unsqueeze(0),
                seg_embeds=self.decoder_input_dropout(inputs[j_start:j_end, :, :]),
            )
            decoder_output = self.embedding2vocab(decoder_output)
            decoder_logpy = F.log_softmax(decoder_output, dim=2)

            decoder_target = x[j_start:j_end, :]

            target_logpy = decoder_logpy[:-1, :, :].gather(dim=2, index=decoder_target.unsqueeze(-1)).squeeze(-1)

            tmp_logpy = torch.zeros_like(target_logpy[0])

            # j is a temporary j_end.
            for j in range(j_start, j_end):
                tmp_logpy = tmp_logpy + target_logpy[j - j_start, :]
                if j > j_start:
                    tmp_logpy = tmp_logpy + is_single[j, :]
                if j == j_start + 1:
                    tmp_logpy = tmp_logpy + is_single[j_start, :]

                logpy[j_start][j - j_start] = tmp_logpy \
                    + decoder_logpy[j - j_start + 1, :, self.config.eos_id] \
                    + (1-impact_matrix[j -1,:])


        if mode == 'unsupervised' or mode == 'supervised':

            # total_log_probability
            # log probability for generate <bos> at beginning is 0
            alpha = neg_inf_vector.repeat(max_length - 1, 1)
            alpha[0] = 0

            for j_end in range(1, max_length - 1):
                logprobs = []
                for j_start in range(max(0, j_end - self.config.max_segment_length), j_end):
                    logprobs.append(alpha[j_start] + logpy[j_start + 1][j_end - j_start - 1])
                alpha[j_end] =  torch.logsumexp(torch.stack(logprobs), dim=0)

            NLL_loss = 0.0
            total_length = 0

            # alphas = torch.stack(alpha)
            alphas = alpha
            index = (torch.LongTensor(lengths) - 2).view(1, -1)

            if alphas.is_cuda:
                index = index.cuda()

            NLL_loss = - torch.gather(input=alphas, dim=0, index=index)

            assert NLL_loss.view(-1).size(0) == batch_size

            total_length += sum(lengths) - 2 * batch_size

            normalized_NLL_loss = NLL_loss.sum() / float(total_length)

            if mode == 'supervised':
                # Get extra loss for supervised segmentation

                supervised_NLL_loss = 0.0
                total_length = 0

                for i in range(batch_size):
                    j_start = 1
                    for j_length in segments[i]:
                        if j_length <= self.config.max_segment_length:
                            supervised_NLL_loss = supervised_NLL_loss - logpy[j_start][j_length - 1][i]
                            total_length += j_length
                        j_start += j_length
                
                normalized_supervised_NLL_loss = supervised_NLL_loss / float(total_length)

                # normalized_NLL_loss = normalized_supervised_NLL_loss * 0.1 + normalized_NLL_loss
                normalized_NLL_loss = normalized_supervised_NLL_loss # + normalized_NLL_loss


            return normalized_NLL_loss

        elif mode == 'decode':
            ret = []

            for i in range(batch_size):
                alpha = [-loginf]*(lengths[i] - 1)
                prev = [-1]*(lengths[i] - 1)
                alpha[0] = 0.0
                for j_end in range(1, lengths[i] - 1):
                    for j_start in range(max(0, j_end - self.config.max_segment_length), j_end):
                        logprob = alpha[j_start] + logpy[j_start + 1][j_end - j_start - 1][i].item()
                        if logprob > alpha[j_end]:
                            alpha[j_end] = logprob
                            prev[j_end] = j_start

                j_end = lengths[i] - 2
                segment_lengths = []
                while j_end > 0:
                    prev_j = prev[j_end]
                    segment_lengths.append(j_end - prev_j)
                    j_end = prev_j

                segment_lengths = segment_lengths[::-1]

                ret.append(segment_lengths)

            return ret

        else:
            raise ValueError('Mode %s not supported' % mode)


    def forward_hope_faster_but_not(self, x, lengths, segments=None, mode='unsupervised'):

        # `x` shape: (seq_len, batch_size)
        x = x.transpose(0, 1).contiguous()

        #transformed format: (seq_len, batch_size)
        max_length = x.size(0)
        batch_size = x.size(1)

        loginf = 1000000.0

        max_length = max(lengths)

        lm_output = self.context_encoder(x.transpose(0, 1).contiguous())
        inputs = lm_output.embeds.transpose(0, 1)
        encoder_output = lm_output.decoder_hidden.transpose(0, 1)

        is_single = -loginf * ((x == self.config.punc_id) | (x == self.config.eng_id) | (x == self.config.num_id)).type_as(x)

        neg_inf_vector = torch.full_like(inputs[0,:,0], -loginf)

        # `logpy` shape: (S, max_seg_len, B)
        logpy = neg_inf_vector.repeat(max_length - 1, self.config.max_segment_length, 1)
        logpy[0][0] = 0

        # Make context encoder and segment decoder have different learning rate
        encoder_output = encoder_output * 0.5

        seg_dec_hiddens = self.segment_decoder.gen_start_symbol_hidden(encoder_output)
        
        max_seg_len = self.config.max_segment_length

        range_by_length = {}
        for seg_len in range(max_seg_len, 0, -1):
            last_index = max_length - seg_len
            if seg_len == max_seg_len:
                first_index = 0
            else:
                first_index = range_by_length[seg_len + 1][1]
            range_by_length[seg_len] = (first_index, last_index)

        for seg_len in range(max_seg_len, 0, -1):
            range_start, range_end = range_by_length[seg_len]

            # (range, B, (n_layer + 1)*d_model)
            start_symbols_and_h = seg_dec_hiddens[range_start:range_end, :, :]
            seg_embeds, seg_target = [], []
            for s_idx in range(range_start + 1, range_end + 1):
                masked_seg = inputs[s_idx: s_idx + seg_len, :, :]
                target_seg = x[s_idx:s_idx + seg_len, :]
                seg_embeds.append(masked_seg)
                seg_target.append(target_seg)

            # `seg_embeds` shape: (max_seg_len, range, B, d_model)
            # `seg_target` shape: (max_seg_len, range, B)
            seg_embeds = torch.stack(seg_embeds, dim=1)
            seg_target = torch.stack(seg_target, dim=1)

            # `start_symbols_and_h` shape: (1, B', (n_layer + 1)*d_model)
            # `seg_embeds` shape: (max_seg_len, B', d_model)    
            start_symbols_and_h = start_symbols_and_h.view(1, -1, start_symbols_and_h.size(-1))
            seg_embeds = seg_embeds.view(seg_len, -1, seg_embeds.size(-1))

            # (range_seg, B', d_model)
            decoder_output = self.segment_decoder(
                seg_start_hidden=start_symbols_and_h,
                seg_embeds=self.decoder_input_dropout(seg_embeds),
            )
            # (max_seg_len+1, range, B, d_model)
            decoder_output = decoder_output.view(
                seg_len + 1, -1, batch_size, inputs.size(-1)
            )

            # `decoder_logpy` shape: (max_seg_len+1, range, B, V)
            decoder_output = self.embedding2vocab(decoder_output)
            decoder_logpy = F.log_softmax(decoder_output, dim=-1)            
            # (max_seg_len, range, B)
            target_logpy = decoder_logpy[:-1, :, :, :].gather(
                dim=3, index=seg_target.unsqueeze(-1)
            ).squeeze(-1)

            # `tmp_logpy` shape: (range, B)
            tmp_logpy = torch.zeros_like(target_logpy[0])
            for k in range(seg_len):
                tmp_logpy += target_logpy[k, :, :]
                tmp_logpy += is_single[range_start:range_end, :]
                tmp_logpy += decoder_logpy[k + 1, :, :, self.config.eos_id]
                logpy[range_start:range_end, k, :] = tmp_logpy

        for j_start in range(1, max_length - 1):
            j_end = j_start + min(self.config.max_segment_length, (max_length-1) - j_start)
            decoder_output = self.segment_decoder(
                seg_start_hidden=seg_dec_hiddens[j_start-1, :, :].unsqueeze(0),
                seg_embeds=self.decoder_input_dropout(inputs[j_start:j_end, :, :]),
            )
            decoder_output = self.embedding2vocab(decoder_output)
            decoder_logpy = F.log_softmax(decoder_output, dim=2)

            decoder_target = x[j_start:j_end, :]

            target_logpy = decoder_logpy[:-1, :, :].gather(dim=2, index=decoder_target.unsqueeze(-1)).squeeze(-1)

            tmp_logpy = torch.zeros_like(target_logpy[0])

            #j is a temporary j_end
            for j in range(j_start, j_end):
                tmp_logpy = tmp_logpy + target_logpy[j - j_start, :]
                if j > j_start:
                    tmp_logpy = tmp_logpy + is_single[j, :]
                if j == j_start + 1:
                    tmp_logpy = tmp_logpy + is_single[j_start, :]
                logpy[j_start][j - j_start] = tmp_logpy + decoder_logpy[j - j_start + 1, :, self.config.eos_id]

        if mode == 'unsupervised' or mode == 'supervised':

            # total_log_probability
            # log probability for generate <bos> at beginning is 0
            alpha = neg_inf_vector.repeat(max_length - 1, 1)
            alpha[0] = 0

            for j_end in range(1, max_length - 1):
                logprobs = []
                for j_start in range(max(0, j_end - self.config.max_segment_length), j_end):
                    logprobs.append(alpha[j_start] + logpy[j_start + 1][j_end - j_start - 1])
                alpha[j_end] =  torch.logsumexp(torch.stack(logprobs), dim=0)

            NLL_loss = 0.0
            total_length = 0

            # alphas = torch.stack(alpha)
            alphas = alpha
            index = (torch.LongTensor(lengths) - 2).view(1, -1)

            if alphas.is_cuda:
                index = index.cuda()

            NLL_loss = - torch.gather(input=alphas, dim=0, index=index)

            assert NLL_loss.view(-1).size(0) == batch_size

            total_length += sum(lengths) - 2 * batch_size

            normalized_NLL_loss = NLL_loss.sum() / float(total_length)

            if mode == 'supervised':
                # Get extra loss for supervised segmentation

                supervised_NLL_loss = 0.0
                total_length = 0

                for i in range(batch_size):
                    j_start = 1
                    for j_length in segments[i]:
                        if j_length <= self.config.max_segment_length:
                            supervised_NLL_loss = supervised_NLL_loss - logpy[j_start][j_length - 1][i]
                            total_length += j_length
                        j_start += j_length

                normalized_supervised_NLL_loss = supervised_NLL_loss / float(total_length)

                # normalized_NLL_loss = normalized_supervised_NLL_loss * 0.1 + normalized_NLL_loss
                normalized_NLL_loss = normalized_supervised_NLL_loss # + normalized_NLL_loss

            return normalized_NLL_loss

        elif mode == 'decode':
            ret = []

            for i in range(batch_size):
                alpha = [-loginf]*(lengths[i] - 1)
                prev = [-1]*(lengths[i] - 1)
                alpha[0] = 0.0
                for j_end in range(1, lengths[i] - 1):
                    for j_start in range(max(0, j_end - self.config.max_segment_length), j_end):
                        logprob = alpha[j_start] + logpy[j_start + 1][j_end - j_start - 1][i].item()
                        if logprob > alpha[j_end]:
                            alpha[j_end] = logprob
                            prev[j_end] = j_start

                j_end = lengths[i] - 2
                segment_lengths = []
                while j_end > 0:
                    prev_j = prev[j_end]
                    segment_lengths.append(j_end - prev_j)
                    j_end = prev_j

                segment_lengths = segment_lengths[::-1]

                ret.append(segment_lengths)

            return ret

        else:
            raise ValueError('Mode %s not supported' % mode)

    def lm_forward(self, x: torch.Tensor):

        lm_output = self.context_encoder(x)
        # shape: (B, S, d_model)
        encoder_output = lm_output.decoder_hidden

        # `logits` shape: (B, S, V)
        logits = F.softmax(self.embedding2vocab(encoder_output), dim=-1)
        shift_logits = logits[:, :-1]
        shift_labels = x[:, 1:]

        loss = F.cross_entropy(shift_logits.reshape(-1, logits.size(-1)), shift_labels.reshape(-1), ignore_index=0)

        return loss

    def mlm_forward(self, x: Tensor, lengths, ratio: float=0.15):

        masked_input_ids = torch.clone(x)
        masked_labels = torch.clone(x)

        mask_idxs = torch.rand(x.size()) <= ratio
        mask_idxs[:, 0] = False
        for idx, length in enumerate(lengths):
            mask_idxs[idx, length-1:].fill_(False)

        masked_input_ids[mask_idxs] = 103

        mlm_loss = self.context_encoder.mlm_forward(x=masked_input_ids, masked_labels=masked_labels)

        return mlm_loss


class ContextEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, layer_number, dropout_rate):
        super(ContextEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=layer_number,
            dropout=dropout_rate
        )
        #num_layers * num_directions, batch, hidden_size
        self.h_init_state = nn.Parameter(torch.zeros(layer_number, 1, hidden_size))
        self.c_init_state = nn.Parameter(torch.zeros(layer_number, 1, hidden_size))
        if self.input_size != self.hidden_size:
            self.embedding2hidden = nn.Linear(input_size, hidden_size)

    def forward(self, rnn_input, lengths, init_states):
        if self.input_size != self.hidden_size:
            rnn_input = self.embedding2hidden(rnn_input)
        self.rnn.flatten_parameters()
        rnn_input = nn.utils.rnn.pack_padded_sequence(rnn_input, lengths)
        output, _ = self.rnn(rnn_input, init_states)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        return output

    def get_init_states(self, batch_size):
        return (self.h_init_state.expand(-1, batch_size, -1).contiguous(),
                self.c_init_state.expand(-1, batch_size, -1).contiguous())

class SegmentDecoder2(nn.Module):
    def __init__(self, hidden_size, output_size, layer_number, dropout_rate):
        super(SegmentDecoder2, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=layer_number,
            dropout=dropout_rate
        )
        if self.hidden_size != self.output_size:
            self.hidden2embedding = nn.Linear(hidden_size, output_size)
        self.output_dropout = nn.Dropout(p=dropout_rate)

    def forward(self, rnn_input, init_states):
        self.rnn.flatten_parameters()
        output, _ = self.rnn(rnn_input, init_states)
        if self.hidden_size != self.output_size:
            output = self.hidden2embedding(output)
        output = self.output_dropout(output)

        return output

