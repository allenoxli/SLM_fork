r"""Word segmentation by using BERT perturbating probe method."""
# %%
import os
import gc
from typing import List
from tqdm import tqdm
from time import time
import subprocess

import argparse

import torch
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizerFast

from codes.dataloader import load_pickle
from codes._cws_tokenizer import CWSHugTokenizer
from codes.dataloader import ClsDataset, InputDataset, OneShotIterator

def dist(x, y):
    return torch.sqrt(((x - y)**2).sum(-1))

# %%

def eval(eval_command, attr, out_path, is_pred=False):

    out = subprocess.Popen(eval_command.split(' '),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    stdout = stdout.decode("utf-8")

    with open(out_path, 'w') as f_out:
        f_out.write(stdout)

    tail_info = stdout.split('\n')[-15:]

    log_info = f'Test results:\n%s' % '\n'.join(tail_info) if is_pred else f'Validation results:\n%s' % '\n'.join(tail_info)
    F_score = 0
    for line in tail_info:
        if line[:len('=== F MEASURE:')] == '=== F MEASURE:':
            F_score = float(line.split('\t')[-1])

    print(f'{attr}: {F_score=}')

    return F_score


def main(model, tk, upper_bound):
    predict_batch_size = 2000
    predict_dataset = InputDataset([f'data/{data}/test.txt'], tk)
    predict_dataloader = DataLoader(
        dataset=predict_dataset,
        shuffle=False,
        batch_size=predict_batch_size,
        num_workers=0,
        collate_fn=InputDataset.padding_collate
    )

    PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))
    DATA_PATH = f'{PROJECT_ROOT_PATH}/data/{data}'
    SCRIPT = f'{PROJECT_ROOT_PATH}/data/score.pl'
    TRAINING_WORDS = f'{DATA_PATH}/words.txt'
    GOLD_TEST = f'{DATA_PATH}/test_gold.txt'

    fout = open(f'{out_dir}/{data}.txt', 'w')

    pertur_bz = 256
    batch_size = 512

    for x_batch, seq_len_batch, uchars_batch, segments_batch, restore_orders in predict_dataloader:
        # bz_stime = time()
        all_sents, all_sents2 = [], []
        for s_id in tqdm(range(0, len(x_batch), batch_size), dynamic_ncols=True):
            input_ids = x_batch[s_id: s_id+batch_size]
            attention_mask = (input_ids != 0).bool()
            lengths = torch.tensor(seq_len_batch[s_id: s_id+batch_size]) - 2
            uchars_cur = uchars_batch[s_id: s_id+batch_size]

            seq_len = input_ids.size(1) - 2
            # shape: (B, (2*S -1), S)
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
            small_batches = [{
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

            # `vec` shape (B, (2*S-1), S, H)
            new_size = (input_ids.size(0), -1, vectors.size(1), vectors.size(2))
            vec = vectors.view(new_size)

            all_dis = []
            for i in range(1, seq_len): # decide whether the i-th character and the (i+1)-th character should be in one word
                d1 = dist(vec[:, 2 * i, i + 1], vec[:, 2 * i - 1, i + 1])
                d2 = dist(vec[:, 2 * i - 2, i], vec[:, 2 * i - 1, i])
                d = (d1 + d2) / 2
                all_dis.append(d)

            # `all_dis` shape: (B, S-3)
            all_dis = torch.stack(all_dis, dim=1)

            # if d > upper_bound, then we combine the two tokens, else if d <= lower_bound then we segment them.
            labels = torch.where(all_dis>=upper_bound, 1, 0)

            # bos, eos and the first token should be zero label. (it doesn't matter.)
            labels = torch.cat([
                torch.zeros(labels.size(0), 2).to(device),
                labels,
                torch.zeros(labels.size(0), 1).to(device)
            ], dim=-1)

            sents = []
            sents2 = []
            for label, ids, length, uchars in zip(labels, input_ids, lengths, uchars_cur):
                sent, word = [], [ids[1]]
                sent2, word2 = [], [uchars[1]]
                for i in range(2, length+2):
                    if label[i] == 0 or i == (length+1):
                        sent.append(''.join(tk.convert_ids_to_tokens(word)))
                        sent2.append(''.join(word2))
                        word = []
                        word2 = []
                    word.append(ids[i])
                    word2.append(uchars[i])

                sent2.append(uchars[-1])
                sents.append(sent)
                sents2.append(sent2)

            all_sents.extend(sents)
            all_sents2.extend(sents2)

            del vectors
            torch.cuda.empty_cache()
            gc.collect()

        for ord_idx in restore_orders:
            line = tk.segment_token.join(all_sents2[ord_idx])
            fout.write(line.replace('<\\n>', '\n'))

        # print(f'iter time: {time() -bz_stime}')

    TEST_OUTPUT = f'{out_dir}/{data}.txt'
    eval_command_slm = f'perl {SCRIPT} {TRAINING_WORDS} {GOLD_TEST} {TEST_OUTPUT}'

    PRED_SCORE = f'{out_dir}/{data}_score.txt'
    f_score = eval(eval_command_slm, 'slm', PRED_SCORE)

    fout.close()

# %%

if __name__ == '__main__':

    start_time = time()
    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='pku')
    parser.add_argument('--device', type=str, default='0')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device       ### only use two GPUs


    hug_name = 'bert-base-chinese'
    model = BertModel.from_pretrained(hug_name)
    tk = CWSHugTokenizer(vocab_file=None, tk_hug_name=hug_name)

    model.resize_token_embeddings(len(tk))
    model.to(device)
    model.eval()

    upper_bound, lower_bound = 8, 6

    data_list = ['msr', 'pku', 'as', 'cityu']
    data_list = [args.data]
    for data in data_list:
        for upper_bound in [8, 12, 10, 14]:
            out_dir = f'./log/perturbed_masking/bound{upper_bound}'
            os.makedirs(out_dir, exist_ok=True)
            main(model, tk, upper_bound)

    print(f'Process time: {time() - start_time}')


# %%
