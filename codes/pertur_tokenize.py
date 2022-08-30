r"""Word segmentation by using BERT perturbating probe method."""
# %%
import os
from typing import List
from tqdm import tqdm
from time import time
import gc

import torch
from transformers import BertModel, BertTokenizerFast

from codes.dataloader import load_pickle


def dist(x, y):
    return torch.sqrt(((x - y)**2).sum(-1))


# %%

@torch.no_grad()
def main(model, tk, txts: List[str], pertur_bz : int = 16):

    device = next(model.parameters()).device

    tk_res = tk(txts, padding=True, max_length=32, truncation=True, return_tensors='pt')
    input_ids, attention_mask = tk_res.input_ids, tk_res.attention_mask
    length = [(line!=0).sum()-2 for line in input_ids]

    # shape: (B, (2*S -1), S)
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
    # small_batches = [ninput_ids[num*pertur_bz : (num+1)*pertur_bz] for num in range(batch_num)]
    small_batches = [
        {
            'input_ids': ninput_ids[num*pertur_bz : (num+1)*pertur_bz].to(device),
            'attention_mask': nattention_mask[num*pertur_bz : (num+1)*pertur_bz].to(device),
        } for num in range(batch_num)]


    # `vectors` shape: (B*(2*S-1), S, H)
    vectors = None
    for input in small_batches:
        if vectors is None:
            vectors = model(**input).last_hidden_state.detach()
            continue
        vectors = torch.cat([vectors, model(**input).last_hidden_state.detach()], dim=0)


    # print(f'{input["input_ids"].size()=}')
    # print(f'{batch_num=}')
    # print(f'{ninput_ids.size()=}')
    # print(f'{vectors.size()=}')

    # `vec` shape (B, (2*S-1), S, H)
    new_size = (input_ids.size(0), -1, vectors.size(1), vectors.size(2))
    vec = vectors.view(new_size)

    all_dis = []
    for i in range(1, seq_len): # decide whether the i-th character and the (i+1)-th character should be in one word
        d1 = dist(vec[:, 2 * i, i + 1], vec[:, 2 * i - 1, i + 1])
        d2 = dist(vec[:, 2 * i - 2, i], vec[:, 2 * i - 1, i])
        d = (d1 + d2) / 2

        all_dis.append(d)
    all_dis = torch.stack(all_dis, dim=1)

    # if d > upper_bound, then we combine the two tokens,
    # if d <= lower_bound then we segment them.
    upper_bound = 12
    lower_bound = 8
    labels = torch.where(all_dis>=upper_bound, 1, 0)
    # b = torch.where((all_dis<upper_bound) & (all_dis >= lower_bound), -100, 0)

    labels = torch.cat([
        torch.zeros(labels.size(0), 2).to(device),
        labels,
        torch.zeros(labels.size(0), 1).to(device)
    ], dim=-1)

    sents = []
    for label, ids, cur_len in zip(labels, input_ids, length):
        sent, word = [], []
        for i in range(1, cur_len+1):
            word.append(ids[i])
            if label[i] == 1:
                sent.append(''.join(tk.convert_ids_to_tokens(word)))
                word = []
        sents.append(sent)

    del vectors
    torch.cuda.empty_cache()
    gc.collect()

    return sents

# %%

if __name__ == '__main__':

    start_time = time()
    txt = [
        '台南殺警案凶嫌林信吾於昨日清晨4時在新竹遭到警方逮捕',
        '明德外役監副典獄長江振亨表示',
    ]

    device = torch.device('cuda')

    hug_name = 'bert-base-chinese'
    model = BertModel.from_pretrained(hug_name)
    tk = BertTokenizerFast.from_pretrained(hug_name)

    tokenizer = CWSHugTokenizer(vocab_file=None, tk_hug_name=hug_name)

    model.to(device)
    model.eval()

    pertur_bz = 128
    batch_size = 256
    # input_files = ['data/msr/unsegmented.txt']
    input_files = ['data/msr/test.txt']
    input_files = ['data/msr/hug_test_dset.pkl']
    for file in input_files:
        if 'pkl' in file:
            fin = load_pickle(file)
            txts = fin['sent_token']
            print(len(fin))
            print('----')
        else:
            fin = open(file, 'r').readlines()
            txts = [line.strip() for line in fin]

        all_sents = []
        for s_id in tqdm(range(0, len(txts), batch_size), dynamic_ncols=True):
            bz_stime = time()
            sents = main(model=model, tk=tk, txts=txts[s_id: s_id+batch_size], pertur_bz=pertur_bz)
            all_sents.extend(sents)
            print(f'iter time: {time() -bz_stime}')
    
        for line in all_sents:
            print(line)

    print(f'Process time: {time() - start_time}')

# %%
import pickle
def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

a = load_pickle('data/msr/hug_test_dset.pkl')
q = a['sent_tokens']

# %%
from torch.utils import data
from codes.dataloader import InputDataset, OneShotIterator
from codes._cws_tokenizer import CWSHugTokenizer

tokenizer = CWSHugTokenizer(vocab_file=None, tk_hug_name=hug_name)

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

# %%
x_batch, seq_len_batch, uchars_batch, segments_batch = next(supervised_data_iterator)

# %%
x_batch
# %%
