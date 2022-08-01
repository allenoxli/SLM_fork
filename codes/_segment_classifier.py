r"""Train a binary classifier.(word boundary or not.)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import AutoModel
from codes._segment_encoder import SegEmbedding, SegmentEncoder
from codes._util import init_module

class ClassifierModel(nn.Module):
    def __init__(
        self,
        embedding: SegEmbedding,
        d_model: int,
        d_ff: int,
        n_layers: int,
        n_heads: int,
        encoder: SegmentEncoder = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if encoder is not None:
            self.encoder = encoder
        else:
            self.embedding = embedding
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=d_ff, nhead=n_heads, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.encoder.apply(init_module)

        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )

        self.pooler.apply(init_module)

    def forward(self, x: Tensor, attention_mask: Tensor):
        if hasattr(self, 'embedding'):
            embeds = self.embedding(x)
            encoder_outputs = self.encoder(src=embeds, src_key_padding_mask=(attention_mask==True))
        else:
            output = self.encoder(x)
            encoder_outputs = output.decoder_hidden * 0.5

        pool_output = self.pooler(encoder_outputs)

        # `pool_output` shape: (B, S, d_model)
        return (pool_output, )


class SegmentClassifier(nn.Module):
    def __init__(
        self,
        embedding: SegEmbedding,
        d_model: int,
        d_ff: int,
        dropout: float,
        n_layers: int,
        n_heads: int,
        model_type: str,
        pad_id: int,
        encoder: SegmentEncoder = None,
        num_labels: int = 2,
        label_smoothing: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pad_id = pad_id
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)

        self.dropout = nn.Dropout(dropout)

        if model_type == 'segment_encoder':
            assert encoder is not None
        else:
            assert encoder is None

        if 'bert' in model_type:
            self.model = AutoModel.from_pretrained(model_type)
            assert d_model == 768
        elif encoder is not None:
            self.model = ClassifierModel(
                embedding=embedding,
                encoder=encoder,
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                n_layers=n_layers,
                n_heads=n_heads,
            )

        self.classifier = nn.Linear(in_features=d_model, out_features=num_labels)
        init_module(self.classifier)


    def forward(self, x: Tensor, labels: Tensor = None):

        attention_mask = (x != self.pad_id).bool()
        model_output = self.model(
            x,
            attention_mask=attention_mask.to(x.device),
        )

        # `sequence_output` shape: (B, S, d_model)
        sequence_output = model_output[0]

        # `logits` shape: (B, S, num_labels)
        logits = self.classifier(self.dropout(sequence_output))
        logits = F.softmax(logits, -1)

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss

        return logits

    def generate_segments(self, x: Tensor, lengths: Tensor, return_confidence: bool = False):
        r"""Generate the segments for segment model or inference."""

        lengths = torch.tensor(lengths)
        # bos and eos.
        lengths -= 2

        # `logits` shape: (B, S, num_labels)
        logits = self(x)

        # `probs` shape: (B, S)
        # `labels` shape: (B, S)
        probs, labels = logits.max(dim=-1)

        confidence = 0
        batch_segments = []
        # Find the end-of-word boundary.
        for seq_len, line, prob in zip(lengths, labels, probs):
            line = line[1:seq_len+1]

            confidence += prob[1:seq_len+1].mean() if line.nelement != 0 else 0

            segment = []
            seg_len = 1
            for i in range(line.size(0)):
                if line[i] == 1 or i == line.size(0) - 1:
                    segment.append(seg_len)
                    seg_len = 1
                    continue
                seg_len += 1

            assert sum(segment) == seq_len

            batch_segments.append(segment)

        if return_confidence == True:
            return batch_segments, confidence/len(batch_segments)

        return batch_segments
