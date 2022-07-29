
import argparse
import subprocess

import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils import data
from time import time

from codes import model
from codes.tokenization import CWSTokenizer
from codes.dataloader import ClsDataset, InputDataset, OneShotIterator
from codes._segment_classifier import SegmentClassifier


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)


start_time  = time()


MODE = 'unsupervised'
DATA = 'cityu'
MAX_SEG_LEN = 3
EXTRA = 'classifier'

DATA_PATH = f'./data/{DATA}'

GOLD_TEST = f'{DATA_PATH}/test_gold.txt'

TRAINING_WORDS = f'{DATA_PATH}/words.txt'

MODEL_PATH = f'models_{EXTRA}/{MODE}-{DATA}-{MAX_SEG_LEN}'
OUTPUT_PATH = 'output'

TEST_OUTPUT = f'{OUTPUT_PATH}/prediction.txt'
TEST_SCORE = f'{OUTPUT_PATH}/score.txt'

CLS_TEST_OUTPUT = f'{OUTPUT_PATH}/cls_prediction.txt'
CLS_TEST_SCORE = f'{OUTPUT_PATH}/cls_score.txt'

args = {'use_cuda': True, 'do_unsupervised': True, 'do_supervised': False, 'do_valid': True, 'do_predict': True, 'unsegmented': ['data/as/unsegmented.txt', 'data/as/test.txt'], 'segmented': None, 'predict_inputs': ['data/as/test.txt'], 'valid_inputs': ['data/as/test.txt'], 'predict_output': 'models/unsupervised-as-3/prediction.txt', 'valid_output': 'models/unsupervised-as-3/valid_prediction.txt', 'max_seq_length': 32, 'vocab_file': 'data/vocab/vocab.txt', 'config_file': 'models/slm_as_3_config.json', 'init_embedding_path': 'data/vocab/embedding.npy', 'init_checkpoint': None, 'save_path': 'models/unsupervised-as-3', 'gradient_clip': 0.1, 'supervised_lambda': 1.0, 'sgd_learning_rate': 16.0, 'adam_learning_rate': 0.005, 'unsupervised_batch_size': 16000, 'supervised_batch_size': 1000, 'valid_batch_size': 500, 'predict_batch_size': 500, 'save_every_steps': 400, 'log_every_steps': 100, 'warm_up_steps': 800, 'train_steps': 4000, 'cpu_num': 4, 'segment_token': '  ', 'english_token': '<ENG>', 'number_token': '<NUM>', 'punctuation_token': '<PUNC>', 'bos_token': '<BOS>', 'eos_token': '</s>', 'do_classifier': True, 'cls_train_steps': 4000, 'cls_batch_size': 6000, 'cls_adam_learning_rate': 0.005, 'cls_d_model': 256, 'cls_dropout': 0.1}
args = argparse.Namespace(**args)
args.cls_predict_output = 'models/unsupervised-as-3/cls_prediction.txt'

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



config_file = f'models_normal/slm_{DATA}_{MAX_SEG_LEN}_config.json'
pad_id = 0
slm_config = model.SLMConfig.from_json_file(config_file)
init_embedding = np.load(args.init_embedding_path)

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

predict_dataset = InputDataset(args.predict_inputs, tokenizer)

predict_dataloader = data.DataLoader(
    dataset=predict_dataset,
    shuffle=False,
    batch_size=args.predict_batch_size,
    num_workers=0,
    collate_fn=InputDataset.padding_collate
)


device = torch.device('cuda')



def eval(eval_command, attr, out_path):
    print(eval_command)
    out = subprocess.Popen(eval_command.split(' '),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    stdout = stdout.decode("utf-8")

    with open(out_path, 'w') as f_out:
        f_out.write(stdout)

    tail_info = '\n'.join(stdout.split('\n')[-15:])
    print(f'===== {attr} =====')
    print('Test evaluation results:\n%s' % tail_info)

    return stdout, tail_info


######################
# SLM part.
######################
# Load pre-train model.

slm_path = f'{MODEL_PATH}/best-checkpoint'
slm.load_state_dict(torch.load(slm_path)['model_state_dict'])
slm.to(device)
slm.eval()

from tqdm import tqdm
all_res = []

fout_slm = open(TEST_OUTPUT, 'w')
all_res_slm = []
with torch.no_grad():
    for x_batch, seq_len_batch, uchars_batch, segments_batch, restore_orders in tqdm(predict_dataloader, dynamic_ncols=True):
        segments_batch_slm = slm(x_batch.to(device), seq_len_batch, mode='decode')
        for i in restore_orders:
            uchars, segments_slm = uchars_batch[i], segments_batch_slm[i]
            fout_slm.write(tokenizer.restore(uchars, segments_slm))
            all_res_slm.append(
                [
                    x_batch[i],
                    uchars,
                    segments_slm,
                ]
            )

fout_slm.close()

eval_command_slm = f'perl data/score.pl {TRAINING_WORDS} {GOLD_TEST} {TEST_OUTPUT}'
slm_stdout, slm_tail = eval(eval_command_slm, 'SLM', TEST_SCORE)

print('\n\n\n')
######################
# CLS part.
######################
# Load pre-train model.
cls_path = f'{MODEL_PATH}/best-cls_checkpoint'
# cls_path = f'{MODEL_PATH}/cls_checkpoint'
# cls_path = f'models_normal/unsupervised-as-4/cls_checkpoint'
cls_model.load_state_dict(torch.load(cls_path)['model_state_dict'])

cls_model.to(device)
cls_model.eval()

fout_cls = open(CLS_TEST_OUTPUT, 'w')
all_res_cls = []
with torch.no_grad():
    for x_batch, seq_len_batch, uchars_batch, segments_batch, restore_orders in tqdm(predict_dataloader, dynamic_ncols=True):
        segments_batch_cls = cls_model.generate_segments(x=x_batch.to(device), lengths=seq_len_batch)
        for i in restore_orders:
            uchars, segments_cls = uchars_batch[i], segments_batch_cls[i]
            fout_cls.write(tokenizer.restore(uchars, segments_cls))
            all_res_cls.append(
                [
                    x_batch[i],
                    uchars,
                    segments_cls,
                ]
            )

fout_cls.close()
eval_command_cls = f'perl data/score.pl {TRAINING_WORDS} {GOLD_TEST} {CLS_TEST_OUTPUT}'
cls_stdout, cls_tail = eval(eval_command_cls, 'CLS', CLS_TEST_SCORE)



print(f'Process time: {time() - start_time}')
