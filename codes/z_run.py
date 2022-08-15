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
from transformers import BertModel, BertTokenizer
tk = BertTokenizer.from_pretrained("bert-base-chinese")
m = BertModel.from_pretrained('bert-base-chinese')




# %%

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

