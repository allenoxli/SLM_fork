# %%
import argparse
import six
import logging
import os
import random
import subprocess

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
from time import time

import model
from tokenization import CWSTokenizer
from dataloader import ClsDataset, InputDataset, OneShotIterator
from _segment_classifier import SegmentClassifier

import pickle
def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)


# %%


DATA = 'as'
DATA_PATH = f'./data/{DATA}'
MAX_SEG_LEN = 3
MODE = 'unsupervised'

MODEL_PATH=f'models/{MODE}-{DATA}-{MAX_SEG_LEN}'

TRAINING_WORDS = f'{DATA_PATH}/words.txt'
UNSEGMENT_DATA = f'{DATA_PATH}/unsegmented.txt'
SEGMENT_DATA = f'{DATA_PATH}/segmented.txt'


TEST_DATA = f'{DATA_PATH}/test.txt'
GOLD_TEST = f'{DATA_PATH}/test_gold.txt'
TEST_OUTPUT = f'{MODEL_PATH}/prediction.txt'
TEST_SCORE = f'{MODEL_PATH}/score.txt'

VALID_DATA = f'{TEST_DATA}'
GOLD_VALID = f'{GOLD_TEST}'
VALID_OUTPUT = f'{MODEL_PATH}/valid_prediction.txt'
VALID_SCORE = f'{MODEL_PATH}/valid_score.txt'

CONFIG_FILE = f'models/slm_"{DATA}"_"{MAX_SEG_LEN}"_config.json'
INIT_EMBEDDING_PATH = f'data/vocab/embedding.npy'
VOCAB_FILE = f'data/vocab/vocab.txt'




args = parse_args()
args.do_unsupervised = True
args.do_valid = True
args.do_predict = True

args.unsegmented = [f'{UNSEGMENT_DATA}', f'{TEST_DATA}']
args.predict_input = TEST_DATA
args.predict_output = TEST_OUTPUT
args.valid_inputs = VALID_DATA
args.valid_output = VALID_OUTPUT
args.vocab_file = VOCAB_FILE
args.config_file = CONFIG_FILE
args.init_embedding_path = INIT_EMBEDDING_PATH
args.save_path = MODEL_PATH

# %%
args = {'use_cuda': True, 'do_unsupervised': True, 'do_supervised': False, 'do_valid': True, 'do_predict': True, 'unsegmented': ['data/as/unsegmented.txt', 'data/as/test.txt'], 'segmented': None, 'predict_inputs': ['data/as/test.txt'], 'valid_inputs': ['data/as/test.txt'], 'predict_output': 'models/unsupervised-as-3/prediction.txt', 'valid_output': 'models/unsupervised-as-3/valid_prediction.txt', 'max_seq_length': 32, 'vocab_file': 'data/vocab/vocab.txt', 'config_file': 'models/slm_as_3_config.json', 'init_embedding_path': 'data/vocab/embedding.npy', 'init_checkpoint': None, 'save_path': 'models/unsupervised-as-3', 'gradient_clip': 0.1, 'supervised_lambda': 1.0, 'sgd_learning_rate': 16.0, 'adam_learning_rate': 0.005, 'unsupervised_batch_size': 16000, 'supervised_batch_size': 1000, 'valid_batch_size': 500, 'predict_batch_size': 500, 'save_every_steps': 400, 'log_every_steps': 100, 'warm_up_steps': 800, 'train_steps': 4000, 'cpu_num': 4, 'segment_token': '  ', 'english_token': '<ENG>', 'number_token': '<NUM>', 'punctuation_token': '<PUNC>', 'bos_token': '<BOS>', 'eos_token': '</s>', 'do_classifier': True, 'cls_train_steps': 4000, 'cls_batch_size': 6000, 'cls_adam_learning_rate': 0.005, 'cls_d_model': 256, 'cls_dropout': 0.1}
args = argparse.Namespace(**args)

# %%

if not args.do_unsupervised and not args.do_supervised and not args.do_predict:
    raise ValueError("At least one of `do_unsupervised`, `do_supervised` or `do_predict' must be True.")

if args.do_unsupervised and not args.unsegmented:
    raise ValueError("Unsupervised learning requires unsegmented data.")

if args.do_supervised and not args.segmented:
    raise ValueError("Supervised learning requires segmented data.")

if args.do_predict and not (args.predict_inputs or not args.predict_output or not args.init_checkpoint):
    raise ValueError("Predicion requires init_checkpoint, inputs, and output.")

if (args.do_unsupervised or args.do_supervised) and args.save_path is None:
    raise ValueError('Where do you want to save your trained model?')

if args.save_path and not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

print(str(args))
print('\n======\n')

tokenizer = CWSTokenizer(
    vocab_file=args.vocab_file,
    max_seq_length=args.max_seq_length,
    segment_token=args.segment_token,
    english_token=args.english_token,
    number_token=args.number_token,
    punctuation_token=args.punctuation_token,
    bos_token=args.bos_token,
    eos_token=args.eos_token
)


logs = []
# %%




# Dataloader. (unsupervised, supervised, valid)

print('Prepare unsupervised dataloader')
unsupervsied_dataset = InputDataset(
    args.unsegmented,
    tokenizer,
    is_training=True,
    batch_token_size=args.unsupervised_batch_size
)
unsupervised_dataloader = data.DataLoader(
    unsupervsied_dataset,
    num_workers=args.cpu_num,
    batch_size=1,
    shuffle=False,
    collate_fn=InputDataset.single_collate
)
unsupervised_data_iterator = OneShotIterator(unsupervised_dataloader)

# %%
print('Prepare validation dataloader')
valid_dataset = InputDataset(args.valid_inputs, tokenizer)
valid_dataloader = data.DataLoader(
    dataset=valid_dataset,
    shuffle=False,
    batch_size=args.valid_batch_size,
    num_workers=0,
    collate_fn=InputDataset.padding_collate
)

# %%

x_batch, seq_len_batch, uchars_batch, segments_batch = next(unsupervised_data_iterator)
idx = 0
print(x_batch[idx])
print(seq_len_batch[idx])
print(uchars_batch[idx])
print(segments_batch[idx])
print('-----------')
print('-----------')

for x_batch, seq_len_batch, uchars_batch, segments_batch, restore_orders in unsupervised_dataloader:
    idx = 0
    print(x_batch[idx])
    print(seq_len_batch[idx])
    print(uchars_batch[idx])
    print(segments_batch[idx])
    print(restore_orders[idx])
    print()
    break



# %%
pad_id = 0
slm_config = model.SLMConfig.from_json_file(args.config_file)
init_embedding = np.load(args.init_embedding_path)


slm_config = model.SLMConfig.from_json_file(f'../{args.config_file}')
init_embedding = np.load(f'../{args.init_embedding_path}')

slm = model.SegmentalLM(
    config=slm_config,
    init_embedding=init_embedding
)

cls_model = SegmentClassifier(
    embedding=slm.embedding,
    d_model=args.cls_d_model,
    d_ff=None,
    dropout=args.cls_dropout,
    n_layers=None,
    n_heads=None,
    model_type='segment_encoder',
    pad_id=pad_id,
    encoder=slm.context_encoder,
    num_labels=2,
)

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
path = 'models/unsupervised-as-3/cls_checkpoint'
# model = torch.load(path))

cls_model.load_state_dict(torch.load(path)['model_state_dict'])
# %%
device = torch.device('cpu')

for x_batch, seq_len_batch, uchars_batch, segments_batch, restore_orders in valid_dataloader:
    input_ids, labels = x_batch, None
    cls_loss = cls_model(x=input_ids.to(device))
    break

# %%
output = load_pickle('../models/unsupervised-as-3/output.pkl')

all_input_ids = output['input_ids']
all_labels = output['labels']
segments = output['segments']

cls_batch_size = 100



all_input_ids = torch.stack(all_input_ids, 0)
all_labels = torch.stack(all_labels, 0)
idx = 10
loss = cls_model(all_input_ids[:idx], all_labels[:idx])

output = cls_model.model.encoder(all_input_ids[:idx])
encoder_outputs = output.decoder_hidden
pool_output = cls_model.model.pooler(encoder_outputs)

pool_output = (pool_output, )
sequence_output = pool_output[0]

logits = cls_model.classifier(cls_model.dropout(sequence_output))

labels = all_labels[:idx].long()

loss = cls_model.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

ans = logits.argmax(dim=-1)

# %%
p = 'data/as/test_gold.txt'
f = open(p, 'r').readlines()
# idx = 109811
# f[idx]
# %%
a= list(''.join(f[idx].split()))
# %%
# from transformers import (BertTokenizerFast)
tokenizer('')
# %%
