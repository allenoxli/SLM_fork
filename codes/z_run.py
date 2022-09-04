# %%
import os

import torch

from codes.pytorch_hug_bert.modeling import BertModelSeg

bert_path = 'bert-base-chinese/pytorch_model.bin'
print(os.path.exists(bert_path))
model_bert = BertModelSeg.from_pretrained('bert-base-chinese', state_dict=torch.load(bert_path))
model_bert.eval()
model_bert.resize_token_embeddings(21131)

m = model_bert

x = torch.randint(0, 21131, (3, 32))
x = torch.tensor([
    [1,2,3,4,5,6,7,8,9,10],
    [1,2,3,4,5,6,7,8,9,10],
    [1,2,3,4,5,6,7,8,9,10],
])
attn_mask = x > 3
o = m(x, attention_mask=attn_mask, output_all_encoded_layers=False)
print(o)


# %%

a = torch.randn(3, 5, 768)
new_x_shape = a.size()[:-1] + (12, 64)
x = a.view(*new_x_shape)
print(x.size())
x = x.permute(0, 2, 1, 3)
print(x.size())


mask = torch.randn(3, 1, 1, 5)

# %%
import numpy as np
import torch

def get_mask(
    seq_len: int,
    seg_len: int = None,
):
    mask = (torch.ones((seq_len, seq_len))) == 1
    for i in range(seq_len):
        for j in range(1, min(seg_len + 1, seq_len - i)):
            mask[i, i + j] = False

    # mask = mask.float().masked_fill(mask == 0, float('-inf'))
    # mask = mask.masked_fill(mask == 1, float(0.0))
    # mask = mask.float().masked_fill(mask == 0, float('-inf'))
    return mask.bool()

seq_len = 10
shape = 'cloze'
# shape = 'subsequent'
seg_len = 4
window = None
mm = get_mask(seq_len=seq_len, seg_len=seg_len)
print(mm.size())
print(mm)


# %%
x = torch.tensor([
    [1,2,3,4,5,6,7,8,9,10],
    [1,2,3,4,5,6,7,8,9,10],
    [1,2,3,4,5,6,7,8,9,10],
])

bz = 3
max_len = 10
seg_len = 4

query, key = torch.randn(bz, 12, max_len, 64), torch.randn(bz, 12, max_len, 64)

attn_weights = torch.matmul(query, key.transpose(-1, -2))

attention_mask = (x < 11)
attention_mask = attention_mask[:, None, None, :].float()
attention_mask = (1.0 - attention_mask) * torch.finfo(attn_weights.dtype).min



query_length, key_length = query.size(-2), key.size(-2)
bias = torch.tril(torch.ones((max_len, max_len), dtype=torch.uint8)).view(1, 1, max_len, max_len)
bias = get_mask(max_len, seg_len)[None, None, :, :]
print(f'{bias.size()=}')
causal_mask = bias[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)
mask_value = torch.finfo(attn_weights.dtype).min

mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
attn_weights = torch.where(causal_mask, attn_weights, mask_value)

import torch.nn as nn
attn_weights = attn_weights + attention_mask
attn_weights = nn.functional.softmax(attn_weights, dim=-1)
# %%
attn_weights[0, 0] == 0


# %%
import torch

from transformers import BertModel, BertTokenizer
def get_mask(
    seq_len: int,
    seg_len: int = None,
):
    mask = (torch.ones((seq_len, seq_len))) == 1
    for i in range(seq_len):
        for j in range(1, min(seg_len + 1, seq_len - i)):
            mask[i, i + j] = False
    # mask = mask.float().masked_fill(mask == 0, float('-inf'))
    # mask = mask.masked_fill(mask == 1, float(0.0))
    # mask = mask.float().masked_fill(mask == 0, float('-inf'))
    return mask.bool()

tk = BertTokenizer.from_pretrained("bert-base-chinese")
m = BertModel.from_pretrained('bert-base-chinese')


inputs = tk(["今天的天氣很好", "今天的天氣"], return_tensors="pt", padding=True)

x = inputs.input_ids
pad_mask = inputs.attention_mask[:, None, :]
mask = get_mask(x.size(1), 4)[None, ...]

pad_mask.size(), mask.size()
# %%
attn_mask = pad_mask & mask
print(attn_mask)
o = m(x, attn_mask, output_attentions=True)


# %%
o.attentions[0].size()
o.attentions[0]

# %%
import torch
import torch.nn as nn
import random

x = torch.tensor([
    [101,1,2,3,4,5,6,102, 0, 0, 0, 0],
    [101,1,2,3,4,5,6,7,8,102, 0, 0],
])
lengths = torch.tensor([8, 10])
print(x.size())

# random mask to count masked_lm_loss
masked_input_ids = torch.clone(x)
masked_labels = torch.clone(x)

ratio = .15
mask_idxs = torch.rand(x.size()) <= ratio
mask_idxs[:, 0] = False
for idx, length in enumerate(lengths):
    mask_idxs[idx, length-1:].fill_(False)

print(mask_idxs)
print(mask_idxs.sum())
masked_input_ids[mask_idxs] = 103


# %%
def mask_word(x, length, mask_idx):
    mask_idx[0] = False
    mask_idx[length-1:] = False

    for i in range(1, length-1):
        if mask_idx[i]:
            x[i] = 103 # id of [mask]
    return x





# %%

hug_name = 'bert-base-chinese'
tokenizer = CWSHugTokenizer(vocab_file=None, tk_hug_name=hug_name)

# %%

import os
import pickle
import random

import torch
from torch.utils import data


from codes.tokenization import CWSTokenizer
from codes._cws_tokenizer import CWSHugTokenizer
from codes.dataloader import ClsDataset, InputDataset, OneShotIterator

supervsied_dataset = InputDataset(
    ['data/msr/segmented.txt'],
    tokenizer,
    is_training=True,
    batch_token_size=8000
)
supervised_dataloader = data.DataLoader(
    supervsied_dataset,
num_workers=4,
    batch_size=1,
    shuffle=False,
    collate_fn=InputDataset.single_collate
)
supervised_data_iterator = OneShotIterator(supervised_dataloader)




def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)


a = load_pickle('data/msr/hug_train_seg_dset.pkl')
a['sent_segments'][0]

# %%

import torch
import numpy as np
from transformers import BertModel, BertTokenizerFast

hug_name = 'bert-base-chinese'
model = BertModel.from_pretrained(hug_name)
tk = BertTokenizerFast.from_pretrained(hug_name)

# input_ids = torch.tensor([
#     [101, 1,2,3,4,5,6,7,8,9,10, 102],
#     [101, 1,2,3,4,5,6,7,0,0,0, 102],
# ])

# %%

txt = [
    '台南殺警案凶嫌林信吾於昨日清晨4時在新竹遭到警方逮捕',
    '明德外役監副典獄長江振亨表示',
]
tk_res = tk(txt, padding=True, return_tensors='pt')
input_ids, attention_mask = tk_res.input_ids, tk_res.attention_mask
input_ids.size(), attention_mask.size()
length = [(line!=0).sum()-2 for line in input_ids]
# %%

# shape: (B, (2*S -1), S)
seq_len = input_ids.size(1) - 2
ninput_ids = input_ids.unsqueeze(1).repeat(1, (2*seq_len -1), 1)
nattention_mask = attention_mask.unsqueeze(1).repeat(1, (2*seq_len -1), 1)

print(ninput_ids.size())
print(nattention_mask.size())


for i in range(seq_len):
    if i > 0:
        ninput_ids[:, 2 * i - 1, i] = 103 # id of [mask]
        ninput_ids[:, 2 * i - 1, i + 1] = 103 # id of [mask]
    ninput_ids[:, 2 * i, i + 1] = 103 # id of [mask]


print(f'{ninput_ids.size()=}')

pertur_bz = 16

batch_num = ninput_ids.size(0) * ninput_ids.size(1)
if batch_num % pertur_bz == 0:
    batch_num = batch_num // pertur_bz
else:
    batch_num = batch_num // pertur_bz + 1


# `ninput_ids` shape: ( B*(2*S-1), S)
ninput_ids = ninput_ids.view(-1, ninput_ids.size(-1))
nattention_mask = nattention_mask.view(-1, nattention_mask.size(-1))
# small_batches = [ninput_ids[num*pertur_bz : (num+1)*pertur_bz] for num in range(batch_num)]
small_batches = [
    {
        'input_ids': ninput_ids[num*pertur_bz : (num+1)*pertur_bz],
        'attention_mask': nattention_mask[num*pertur_bz : (num+1)*pertur_bz],
    } for num in range(batch_num)]

print(f'{batch_num=}')
print(f'{ninput_ids.size()=}')
print(f'{nattention_mask.size()=}')


# %%
# `vectors` shape: (B*(2*S-1), S, H)
vectors = None
for num, input in enumerate(small_batches):
    if vectors is None:
        vectors = model(**input).last_hidden_state.detach()
        continue
    vectors = torch.cat([vectors, model(**input).last_hidden_state.detach()], dim=0)

print(vectors.size())

# %%

# `vec` shape (B, (2*S-1), S, H)
new_size = (input_ids.size(0), -1, vectors.size(1), vectors.size(2))
vec = vectors.view(new_size)
print(vec.size())

def dist(x, y):
    return torch.sqrt(((x - y)**2).sum(-1))


all_dis = []
for i in range(1, seq_len): # decide whether the i-th character and the (i+1)-th character should be in one word
    d1 = dist(vec[:, 2 * i, i + 1], vec[:, 2 * i - 1, i + 1])
    d2 = dist(vec[:, 2 * i - 2, i], vec[:, 2 * i - 1, i])
    d = (d1 + d2) / 2

    all_dis.append(d)
all_dis = torch.stack(all_dis, dim=1)
print(all_dis.size())

# if d > upper_bound, then we combine the two tokens,
# if d <= lower_bound then we segment them.
upper_bound = 12
lower_bound = 8
a = torch.where(all_dis>=upper_bound, 1, 0)
b = torch.where((all_dis<upper_bound) & (all_dis >= lower_bound), -100, 0)

c = a+b
labels = torch.cat([torch.zeros(c.size(0), 2), c, torch.zeros(c.size(0), 1)], dim=-1)
print(c.size())
print(d.size())


# %%
labels = []
for line in all_dis:
    tmp = [0, 0]
    for d in line:
        if d >= 12:
            tmp.append(1)
        elif d >= 8:
            tmp.append(-100) # -100 is ignored in CrossEntropyLoss()
        else:
            tmp.append(0)
    tmp.append(0)
    labels.append(tmp)


# %%
sents = []
for label, ids, cur_len in zip(d, input_ids, length):
    sent = []
    tmp = []
    for i in range(1, cur_len+1):
        tmp.append(ids[i])
        if label[i] == 1:
            sent.append(tk.decode(tmp))
            tmp = []
    sents.append(sent)

print(sents)

# %%

from transformers import GPT2TokenizerFast, GPT2LMHeadModel
tk = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

s = ['apple is great']
tk()


def beam_search(
    model,
    tk,
    txts: List[str],
    input_ids: torch.LongTensor,
    beam_size: int = 4,
    pad_id: int = 50256,
    bos_id: int = 50256,
    eos_id: int = 50256,
    max_len: int = 32,
):
    device = next(model.parameters()).device
    batch_size = input_ids.size(0)


    batch_tgt_ids = tk(txts, return_tensors='pt').to(device)
    
    beam_scores = torch.zeros(batch_size, beam_size)
    # beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * beam_size,))
    tmp_beam_scores = torch.zeros(batch_size * beam_size * beam_size)

    vocab_size = tk.vocab_size

    # (B, 1)
    batch_tgt_ids = torch.tensor([bos_id] * batch_size)

    for i in range(1, max_len):
        outputs = model(batch_tgt_ids)

        # (B, V)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_logits = F.softmax(next_token_logits, dim=-1)

        if i == 1:
            # (B, num_beams)
            next_token_scores, next_token_ids = next_token_logits.topk(beam_size, dim=-1,  largest=True, sorted=True)
            beam_scores[:batch_size * beam_size] = next_token_scores.view(-1)
            tmp_beam_scores[:batch_size * beam_size] = next_token_scores.view(-1)
            # (B, 1+1)
            batch_tgt_ids = torch.cat([batch_tgt_ids, next_token_ids], dim=-1)
            continue

        next_token_scores, next_token_ids = next_token_logits.topk(beam_size, dim=-1,  largest=True, sorted=True)

        tmp_beam_scores[:batch_size * beam_size]
        torch.cat([])



        # (B, num_beams * V)
        next_token_scores = next_token_scores.view(batch_size, beam_size * vocab_size)

        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * beam_size, dim=1, largest=True, sorted=True
        )
        
        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size


def beam_search(
    model,
    tk,
    input_ids: torch.LongTensor,
    beam_size: int = 4,
    pad_id: int = 50256,
    bos_id: int = 50256,
    eos_id: int = 50256,
    max_len: int = 32,
):
    batch_size = input_ids.size(0)
    
    beam_scores = torch.zeros(batch_size, beam_size)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * beam_size,))

    vocab_size = tk.vocab_size

    # outputs = [[tk.bos_token_id]]
    tgt_ids = []
    for _ in range(max_len):
        outputs = model(tgt_ids)
        # (B, V)
        next_token_logits = outputs.logits[:, -1, :]

        # (B * num_beams, V)
        next_token_scores = F.log_softmax(next_token_logits, dim=-1)

        # (B, num_beams * V)
        next_token_scores = next_token_scores.view(batch_size, beam_size * vocab_size)

        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * beam_size, dim=1, largest=True, sorted=True
        )
        
        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size
    



from typing import Optional, Tuple, List

class BeamSearchScorer:

    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[bool] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        **kwargs,
    ):
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
            )
            for _ in range(batch_size)
        ]
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        beam_indices: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        if not (batch_size == (input_ids.shape[0] // self.group_size)):
            if self.num_beam_groups > 1:
                raise ValueError(
                    f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                    f"size of {self.group_size} is expected by the beam scorer."
                )
            else:
                raise ValueError(
                    f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                    f"{self.group_size} is expected by the beam scorer."
                )

        device = input_ids.device
        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                if self.num_beams < len(beam_hyp):
                    raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    if beam_indices is not None:
                        beam_index = beam_indices[batch_beam_idx]
                        beam_index = beam_index + (next_index,)
                    else:
                        beam_index = None

                    beam_hyp.add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                        beam_indices=beam_index,
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id:"
                    f" {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )
    


class BeamHypotheses:
    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        return len(self.beams)

    def add(self, hyp: torch.LongTensor, sum_logprobs: float, beam_indices: Optional[torch.LongTensor] = None):
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, beam_indices))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self.worst_score >= cur_score
            return ret

# %%
a = torch.tensor([[1,2,3,4], [5,6,7,8]])
score, idx = a.topk(2, -1)

print(score)
# print(a[idx.squeeze()])
# %%
torch.gather(input=a, dim=1, index=idx)


# %%
torch.scatter(src=a, dim=1, index=idx)


# %%
from codes import model
config_file = 'models/slm_as_4_config_bert_seg.json'
config = model.SLMConfig.from_json_file(config_file)
# %%

from transformers import AutoModel, BertToken
name = 'hfl/chinese-roberta-wwm-ext'
model = AutoModel.from_pretrained(name)

model


# %%
from transformers import AutoTokenizer, AutoModelForMaskedLM
tk = AutoTokenizer.from_pretrained("bert-base-chinese")
# %%
ids = tk('我們，你們').input_ids
print(ids)
print(tk.convert_ids_to_tokens(ids))


# %%
tk_hfl = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

# %%
