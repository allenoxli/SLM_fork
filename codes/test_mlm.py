import os

import torch
import torch.nn as nn
from transformers import AutoModel, BertForMaskedLM, AutoTokenizer
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from codes.dataloader import InputDataset, OneShotIterator
from codes._cws_tokenizer import CWSHugTokenizer
from codes._util import set_seed

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

set_seed(42)

device = torch.device('cuda')

hug_name = 'bert-base-chinese'


mlm_model = BertForMaskedLM.from_pretrained(hug_name)
tk = CWSHugTokenizer(
    vocab_file=None,
    tk_hug_name=hug_name,
    max_seq_length=32,
)

mlm_model.train()
mlm_model.to(device)


input_file = ['data/msr/unsegmented.txt']
batch_size = 2000
dset = InputDataset(
    input_file,
    tk,
    is_training=True,
    batch_token_size=batch_size
)
dldr = DataLoader(
    dset,
    num_workers=4,
    batch_size=1,
    shuffle=False,
    collate_fn=InputDataset.single_collate
)
data_iterator = OneShotIterator(dldr)


train_steps = 2000
mask_ratio = 0.15

writer = SummaryWriter('models_mlm')

tqdm_step = tqdm(range(train_steps), dynamic_ncols=True)


mlm_model.resize_token_embeddings(len(tk))

# optimizer = AdamW(mlm_model.parameters(), lr=1e-4)
optimizer = Adam(mlm_model.parameters(), lr=5e-4)


for step in tqdm_step:
    x_batch, seq_len_batch, uchars_batch, segments_batch = next(data_iterator)
    x_batch = x_batch.to(device)
    masked_input_ids = torch.clone(x_batch)
    masked_labels = torch.clone(x_batch)

    mask_idxs = torch.rand(x_batch.size()) <= mask_ratio
    mask_idxs[:, 0] = False
    for idx, length in enumerate(seq_len_batch):
        mask_idxs[idx, length-1:].fill_(False)

    masked_input_ids[mask_idxs==True] = 103 # mask index.
    masked_labels[mask_idxs==False] = 0  # ignored index.

    attention_mask = (x_batch != tk.pad_id).bool().to(x_batch.device)

    output = mlm_model(
        input_ids=masked_input_ids,
        attention_mask=attention_mask,
        # labels=masked_labels,
    )
    # loss = output.loss
    logits = output.logits

    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), masked_labels.reshape(-1), ignore_index=0)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    mlm_model.zero_grad()

    tqdm_step.set_description(f'mlm loss: {loss:.3f}')

    writer.add_scalar('mlm loss', loss, step)




