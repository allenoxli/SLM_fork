#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SLM Training and Decoding
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

from codes import model
from codes.tokenization import CWSTokenizer
from codes.dataloader import ClsDataset, InputDataset, OneShotIterator
from codes._segment_classifier import SegmentClassifier
from codes._util import load_pickle, save_pickle, set_logger, set_seed


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Training and decoding SLM (segmental language model)",
        usage="train.py [<args>] [-h | --help]"
    )

    parser.add_argument("--use_cuda", action='store_true', help="Whether to use gpu.")

    # mode
    parser.add_argument("--do_unsupervised", action='store_true', help="Whether to run unsupervised training.")
    parser.add_argument("--do_supervised", action='store_true', help="Whether to run supervised training.")
    parser.add_argument("--do_valid", action='store_true', help="Whether to do validation during training.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run prediction.")

    # general setting
    parser.add_argument("--unsegmented", type=str, nargs="+", help="Path of unsegmented input file")

    parser.add_argument("--segmented", type=str, nargs="+", help="Path of segmented input file")

    parser.add_argument("--predict_inputs", type=str, nargs='+', help="Path to prediction input file")
    parser.add_argument("--valid_inputs", type=str, nargs='+', help="Path to validation input file")
    parser.add_argument("--predict_output", type=str, help="Path to prediction result")
    parser.add_argument("--valid_output", type=str, help="Path to validation output file")

    parser.add_argument("--max_seq_length", type=int, default=32, help="The maximum input sequence length")

    parser.add_argument("--vocab_file", type=str, required=True, help="Path to vocabulary")
    parser.add_argument("--config_file", type=str, required=True, help="Path to SLM configuration file")

    parser.add_argument("--init_embedding_path", type=str, default=None, help="Path to init word embedding")
    parser.add_argument("--init_checkpoint", type=str, default=None, help="Path to init checkpoint")
    parser.add_argument("--save_path", type=str, help="Path to saving checkpoint")

    # training setting
    parser.add_argument("--gradient_clip", type=float, default=0.1)
    parser.add_argument("--supervised_lambda", type=float, default=1.0, help="supervised weight, total_loss = unsupervised_loss + supervised_lambda * supervised_loss")
    parser.add_argument("--sgd_learning_rate", type=float, default=16.0)
    parser.add_argument("--adam_learning_rate", type=float, default=0.005)

    parser.add_argument("--unsupervised_batch_size", type=int, default=6000)
    parser.add_argument("--supervised_batch_size", type=int, default=1000)
    parser.add_argument("--valid_batch_size", type=int, default=16)
    parser.add_argument("--predict_batch_size", type=int, default=16)

    # training step setting
    parser.add_argument("--save_every_steps", type=int, default=400)
    parser.add_argument("--log_every_steps", type=int, default=100)
    parser.add_argument("--warm_up_steps", type=int, default=800)
    parser.add_argument("--train_steps", type=int, default=4000)

    # other setting
    parser.add_argument("--cpu_num", type=int, default=4)
    parser.add_argument("--segment_token", type=str, default='  ', help="Segment token")
    parser.add_argument("--english_token", type=str, default='<ENG>', help="token for English characters")
    parser.add_argument("--number_token", type=str, default='<NUM>', help="token for numbers")
    parser.add_argument("--punctuation_token", type=str, default='<PUNC>', help="token for punctuations")
    parser.add_argument("--bos_token", type=str, default='<BOS>', help="token for begin of sentence")
    parser.add_argument("--eos_token", type=str, default='</s>', help="token for begin of sentence")

    # Classifier arguments.
    parser.add_argument("--do_classifier", action='store_true', help="Whether to run classification.")
    parser.add_argument("--cls_train_steps", type=int, default=4000)
    parser.add_argument("--cls_batch_size", type=int, default=6000)
    parser.add_argument("--cls_adam_learning_rate", type=float, default=0.005)
    parser.add_argument("--cls_d_model", type=int, default=256)
    parser.add_argument("--cls_dropout", type=int, default=0.1)

    parser.add_argument("--iterative_train", action='store_true', help="Whether to run classification.")
    parser.add_argument("--iterative_train_steps", type=int, default=2000)


    parser.add_argument("--seed", type=int, default=42)


    return parser.parse_args(args)


def eval(eval_command, attr, out_path, is_pred=False):
    logging.info(eval_command)
    out = subprocess.Popen(eval_command.split(' '),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    stdout = stdout.decode("utf-8")

    with open(out_path, 'w') as f_out:
        f_out.write(stdout)

    tail_info = stdout.split('\n')[-15:]

    logging.info(f'===== {attr} =====')

    log_info = f'Test results:\n%s' % '\n'.join(tail_info) if is_pred else f'Validation results:\n%s' % '\n'.join(tail_info)
    logging.info(log_info)

    F_score = 0
    for line in tail_info:
        if line[:len('=== F MEASURE:')] == '=== F MEASURE:':
            F_score = float(line.split('\t')[-1])

    print(f'{attr}: {F_score=}')

    return F_score


def main(args):
    device, tokenizer, slm_config, slm = first_part(args)

    logs = []

    print('======')
    print(f'{args.save_path=}')
    print('======')

    PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))


    MODE, DATA, MAX_SEG_LEN = args.save_path.split('/')[-1].rsplit('-')
    DATA_PATH = f'{PROJECT_ROOT_PATH}/data/{DATA}'

    # score script, words and gold.
    SCRIPT = f'{PROJECT_ROOT_PATH}/data/score.pl'
    TRAINING_WORDS = f'{DATA_PATH}/words.txt'
    GOLD_TEST = f'{DATA_PATH}/test_gold.txt'

    MODEL_DIR = args.save_path.split('/')[0]

    MODEL_PATH = f'{PROJECT_ROOT_PATH}/{MODEL_DIR}/{MODE}-{DATA}-{MAX_SEG_LEN}'

    # score and predcition output file.
    # slm valid.
    VALID_OUTPUT = f'{MODEL_PATH}/valid_prediction.txt'
    VALID_SCORE = f'{MODEL_PATH}/valid_score.txt'

    # slm prediction.
    TEST_OUTPUT = f'{MODEL_PATH}/prediction.txt'
    TEST_SCORE = f'{MODEL_PATH}/score.txt'

    # cls valid.
    CLS_VALID_OUTPUT = f'{MODEL_PATH}/valid_prediction_cls.txt'
    CLS_VALID_SCORE = f'{MODEL_PATH}/valid_score_cls.txt'

    # cls prediction.
    CLS_TEST_OUTPUT = f'{MODEL_PATH}/prediction_cls.txt'
    CLS_TEST_SCORE = f'{MODEL_PATH}/score_cls.txt'

    if args.do_unsupervised or args.do_supervised:

        # Optimizer and scheduler.
        adam_optimizer = optim.Adam(slm.parameters(), lr=args.adam_learning_rate, betas=(0.9, 0.998))
        lr_lambda = lambda step: 1 if step < 0.8 * args.train_steps else 0.1
        scheduler = optim.lr_scheduler.LambdaLR(adam_optimizer, lr_lambda=lr_lambda)

        # Dataloader. (unsupervised, supervised, valid)
        if args.do_unsupervised:
            logging.info('Prepare unsupervised dataloader')
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

        if args.do_supervised:
            logging.info('Prepare supervised dataloader')
            supervsied_dataset = InputDataset(
                args.segmented,
                tokenizer,
                is_training=True,
                batch_token_size=args.supervised_batch_size
            )
            supervised_dataloader = data.DataLoader(
                supervsied_dataset,
                num_workers=args.cpu_num,
                batch_size=1,
                shuffle=False,
                collate_fn=InputDataset.single_collate
            )
            supervised_data_iterator = OneShotIterator(supervised_dataloader)

        if args.do_valid:
            logging.info('Prepare validation dataloader')
            valid_dataset = InputDataset(args.valid_inputs, tokenizer)
            valid_dataloader = data.DataLoader(
                dataset=valid_dataset,
                shuffle=False,
                batch_size=args.valid_batch_size,
                num_workers=0,
                collate_fn=InputDataset.padding_collate
            )


        if args.iterative_train:
            # pass
            logging.info('Initializing classifier model parameters.')
            pad_id = 0
            cls_model = SegmentClassifier(
                embedding=None, #slm.embedding,
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
            cls_model.to(device)
            cls_model.train()

            cls_adam_optimizer = optim.Adam(cls_model.parameters(), lr=args.cls_adam_learning_rate, betas=(0.9, 0.998))
            lr_lambda = lambda step: 1 if step < 0.8 * args.cls_train_steps else 0.1
            cls_scheduler = optim.lr_scheduler.LambdaLR(cls_adam_optimizer, lr_lambda=lr_lambda)


        logging.info('Ramdomly Initializing SLM parameters...')
        global_step = 0
        best_F_score = 0
        best_F_score_cls = 0

        # SLM traiing.
        # Start to training.
        for step in range(global_step, args.train_steps):

            slm.train()
            # cls part.
            if args.iterative_train and step > args.iterative_train_steps:
                cls_model.train()

            log = {}

            if args.do_unsupervised:
                x_batch, seq_len_batch, uchars_batch, segments_batch = next(unsupervised_data_iterator)
                x_batch = x_batch.to(device)
                loss = slm(x_batch, seq_len_batch, mode='unsupervised')
                log['unsupervised_loss'] = loss.item()

                # cls part.
                if args.iterative_train and step > args.iterative_train_steps:
                    with torch.no_grad():
                        segments_batch_slm = slm(x_batch, seq_len_batch, mode='decode')
                        batch_labels = []
                        for input_ids, segment in zip(x_batch, segments_batch_slm):
                            label = torch.zeros(input_ids.size(0))
                            e_idx = (input_ids == 5).nonzero(as_tuple=True)[0]
                            label[e_idx+1:].fill_(-100)
                            # `label` including the labels of <bos> and <eos> token.
                            # `-1+1` for correct segment index and BOS token.
                            idx = [sum(segment[:i])-1+1 for i in range(1, len(segment)+1)]
                            label[idx] = 1
                            batch_labels.append(label)

                    labels = torch.stack(batch_labels, 0).long()
                    cls_loss = cls_model(x=x_batch, labels=labels.to(device))
                    log['cls_loss'] = cls_loss.item()

                    loss += cls_loss

            elif args.do_supervised:
                x_batch, seq_len_batch, uchars_batch, segments_batch = next(supervised_data_iterator)
                x_batch = x_batch.to(device)
                loss = slm(x_batch, seq_len_batch, segments_batch, mode='supervised')
                log['supervised_loss'] = loss.item()

            logs.append(log)

            loss.backward()
            nn.utils.clip_grad_norm_(slm.parameters(), args.gradient_clip)

            if step > args.warm_up_steps:
                adam_optimizer.step()
            else:
                # do manually SGD
                for p in slm.parameters():
                    if p.grad is not None:
                        p.data.add_(-args.sgd_learning_rate, p.grad.data)

            scheduler.step()
            slm.zero_grad()
            adam_optimizer.zero_grad()

            # cls part.
            if args.iterative_train and step > args.iterative_train_steps:
                nn.utils.clip_grad_norm_(cls_model.parameters(), args.gradient_clip)
                cls_adam_optimizer.step()
                # if step > args.warm_up_steps:
                #     cls_adam_optimizer.step()
                # else:
                #     # do manually SGD
                #     for p in cls_model.parameters():
                #         if p.grad is not None:
                #             p.data.add_(-args.sgd_learning_rate, p.grad.data)

                cls_scheduler.step()
                cls_model.zero_grad()
                cls_adam_optimizer.zero_grad()

            if step % args.log_every_steps == 0:
                logging.info("global_step = %s" % step)
                if len(logs) > 0:
                    for key in logs[0]:
                        logging.info("%s = %f" % (key, sum([log[key] for log in logs])/len(logs)))
                else:
                    logging.info("Currently no metrics available")
                logs = []

            if (step % args.save_every_steps == 0) or (step == args.train_steps - 1):
                logging.info('Saving checkpoint %s...' % args.save_path)
                slm_config.to_json_file(os.path.join(args.save_path, 'config.json'))
                torch.save({
                    'global_step': step,
                    'best_F_score': best_F_score,
                    'model_state_dict': slm.state_dict(),
                    'adam_optimizer': adam_optimizer.state_dict()
                }, os.path.join(args.save_path, 'checkpoint'))
                if args.iterative_train and step > args.iterative_train_steps:
                    torch.save({
                        'global_step': step,
                        'best_F_score': best_F_score_cls,
                        'model_state_dict': cls_model.state_dict(),
                        'adam_optimizer': cls_adam_optimizer.state_dict()
                    }, os.path.join(args.save_path, 'cls_checkpoint'))

                if args.do_valid:
                    slm.eval()
                    if args.iterative_train and step > args.iterative_train_steps:
                        cls_model.eval()

                    fout_slm = open(VALID_OUTPUT, 'w')
                    fout_cls = open(CLS_VALID_OUTPUT, 'w')

                    with torch.no_grad():
                        for x_batch, seq_len_batch, uchars_batch, segments_batch, restore_orders in valid_dataloader:
                            x_batch = x_batch.to(device)
                            segments_batch_slm = slm(x_batch, seq_len_batch, mode='decode')
                            # cls part.
                            if args.iterative_train and step > args.iterative_train_steps:
                                segments_batch_cls = cls_model.generate_segments(x=x_batch, lengths=seq_len_batch)
                            for i in restore_orders:
                                uchars, segments_slm = uchars_batch[i], segments_batch_slm[i]
                                fout_slm.write(tokenizer.restore(uchars, segments_slm))
                                # cls part.
                                if args.iterative_train and step > args.iterative_train_steps:
                                    segments_cls = segments_batch_cls[i]
                                    fout_cls.write(tokenizer.restore(uchars, segments_cls))

                    fout_slm.close()
                    fout_cls.close()

                    # eval_command_slm = "bash run.sh valid %s" % ' '.join(args.save_path.split('/')[-1].rsplit('-'))
                    # eval_command_cls = "bash run.sh valid_cls %s" % ' '.join(args.save_path.split('/')[-1].rsplit('-'))

                    eval_command_slm = f'perl {SCRIPT} {TRAINING_WORDS} {GOLD_TEST} {VALID_OUTPUT}'
                    eval_command_cls = f'perl {SCRIPT} {TRAINING_WORDS} {GOLD_TEST} {CLS_VALID_OUTPUT}'

                    F_score = eval(eval_command_slm, 'slm', VALID_SCORE)
                    if args.iterative_train and step > args.iterative_train_steps:
                        F_score_cls = eval(eval_command_cls, 'cls', CLS_VALID_SCORE)

                if (not args.do_valid) or (F_score > best_F_score):
                    best_F_score = F_score
                    logging.info('Overwriting best checkpoint....')
                    os.system('cp %s %s' % (os.path.join(args.save_path, 'checkpoint'),
                                            os.path.join(args.save_path, 'best-checkpoint')))
                # cls part.
                if args.iterative_train and step > args.iterative_train_steps:
                    if (F_score_cls > best_F_score_cls):
                        best_F_score_cls = F_score_cls
                        logging.info('Overwriting best cls-checkpoint....')
                        os.system('cp %s %s' % (os.path.join(args.save_path, 'cls_checkpoint'),
                                                os.path.join(args.save_path, 'best-cls_checkpoint')))

        if args.do_classifier:
            logging.info('Initializing classifier model parameters.')
            pad_id = 0
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
            cls_model.to(device)
            cls_model.train()

            cls_adam_optimizer = optim.Adam(cls_model.parameters(), lr=args.cls_adam_learning_rate, betas=(0.9, 0.998))
            lr_lambda = lambda step: 1 if step < 0.8 * args.cls_train_steps else 0.1
            cls_scheduler = optim.lr_scheduler.LambdaLR(cls_adam_optimizer, lr_lambda=lr_lambda)

        # 先做完 slm 再進行 cls.
        if args.do_classifier:
            logging.info('==============================')
            logging.info('==============================')
            logging.info('====== Classifier Task ======')
            logging.info('==============================')
            logging.info('==============================')
            slm.eval()

            # Generate Pseudo labeling.
            all_input_ids, all_labels = [], []
            all_segments = []
            with torch.no_grad():
                for x_batch, seq_len_batch, uchars_batch, segments_batch, restore_orders in valid_dataloader:
                    segments_batch_slm = slm(x_batch.to(device), seq_len_batch, mode='decode')
                    batch_labels = []
                    for input_ids, segment in zip(x_batch, segments_batch_slm):
                        label = torch.zeros(input_ids.size(0))
                        e_idx = (input_ids == 5).nonzero(as_tuple=True)[0]
                        label[e_idx+1:].fill_(-100)

                        # `label` including the labels of <bos> and <eos> token.
                        # `-1+1` for correct segment index and BOS token.
                        idx = [sum(segment[:i])-1+1 for i in range(1, len(segment)+1)]

                        label[idx] = 1
                        batch_labels.append(label)

                    all_input_ids.extend(x_batch)
                    all_labels.extend(batch_labels)

            save_pickle({
                'input_ids':all_input_ids,
                'labels': all_labels,
                'segments': all_segments,

            }, f'{args.save_path}/output.pkl')

            print('------- finish cls labels ------')

            cls_dset = ClsDataset(
                input_ids=all_input_ids,
                pseudo_labels=all_labels,
                real_labels=all_labels,
                is_training=True,
                batch_token_size=args.cls_batch_size
            )

            cls_dldr = data.DataLoader(
                dataset=cls_dset,
                num_workers=args.cpu_num,
                batch_size=1,
                shuffle=False,
                collate_fn=ClsDataset.single_collate
            )
            cls_data_iterator = OneShotIterator(cls_dldr)


            logs = []
            log = {}

            print('------------- Start to Training CLS model -------------')
            for step in range(args.cls_train_steps):
                slm.train()
                cls_model.train()

                input_ids, labels = next(cls_data_iterator)
                cls_loss = cls_model(x=input_ids.to(device), labels=labels.to(device))
                log['cls_loss'] = cls_loss.item()

                logs.append(log)

                cls_loss.backward()
                nn.utils.clip_grad_norm_(cls_model.parameters(), args.gradient_clip)

                if step > args.warm_up_steps:
                    cls_adam_optimizer.step()
                else:
                    # do manually SGD
                    for p in cls_model.parameters():
                        if p.grad is not None:
                            p.data.add_(-args.sgd_learning_rate, p.grad.data)
                    cls_adam_optimizer.step()
                cls_scheduler.step()
                cls_model.zero_grad()
                cls_adam_optimizer.zero_grad()

                if step % args.log_every_steps == 0:
                    logging.info("cls global_step = %s" % step)
                    if len(logs) > 0:
                        for key in logs[0]:
                            logging.info("%s = %f" % (key, sum([log[key] for log in logs])/len(logs)))
                    else:
                        logging.info("Currently no metrics available")
                    logs = []

                if (step % args.save_every_steps == 0) or (step == args.cls_train_steps - 1):
                    logging.info('Saving cls checkpoint %s...' % args.save_path)
                    torch.save({
                        'global_step': step,
                        'best_F_score': best_F_score_cls,
                        'model_state_dict': cls_model.state_dict(),
                        'adam_optimizer': cls_adam_optimizer.state_dict()
                    }, os.path.join(args.save_path, 'cls_checkpoint'))

                    if args.do_valid:
                        slm.eval()
                        cls_model.eval()

                        fout_slm = open(VALID_OUTPUT, 'w')
                        fout_cls = open(CLS_VALID_OUTPUT, 'w')

                        with torch.no_grad():
                            for x_batch, seq_len_batch, uchars_batch, segments_batch, restore_orders in valid_dataloader:
                                x_batch = x_batch.to(device)
                                segments_batch_slm = slm(x_batch, seq_len_batch, mode='decode')
                                segments_batch_cls = cls_model.generate_segments(x=x_batch, lengths=seq_len_batch)
                                for i in restore_orders:
                                    uchars, segments_slm, segments_cls = uchars_batch[i], segments_batch_slm[i], segments_batch_cls[i]
                                    fout_slm.write(tokenizer.restore(uchars, segments_slm))
                                    fout_cls.write(tokenizer.restore(uchars, segments_cls))

                        fout_slm.close()
                        fout_cls.close()

                        # eval_command_slm = "bash run.sh valid %s" % ' '.join(args.save_path.split('/')[-1].rsplit('-'))
                        # eval_command_cls = "bash run.sh valid_cls %s" % ' '.join(args.save_path.split('/')[-1].rsplit('-'))
                        eval_command_slm = f'perl {SCRIPT} {TRAINING_WORDS} {GOLD_TEST} {VALID_OUTPUT}'
                        eval_command_cls = f'perl {SCRIPT} {TRAINING_WORDS} {GOLD_TEST} {CLS_VALID_OUTPUT}'

                        F_score = eval(eval_command_slm, 'slm', VALID_SCORE)
                        F_score_cls = eval(eval_command_cls, 'cls', CLS_VALID_SCORE)

                    if F_score > best_F_score:
                        best_F_score = F_score
                        logging.info('Overwriting best slm checkpoint.... (Train cls model)')
                        os.system('cp %s %s' % (os.path.join(args.save_path, 'checkpoint'),
                                                os.path.join(args.save_path, 'best-checkpoint')))

                    if F_score_cls > best_F_score_cls:
                        best_F_score_cls = F_score_cls
                        logging.info('Overwriting best cls checkpoint.... (Train cls model)')
                        os.system('cp %s %s' % (os.path.join(args.save_path, 'cls_checkpoint'),
                                                os.path.join(args.save_path, 'best-cls_checkpoint')))

    if args.do_predict:
        logging.info('Prepare prediction dataloader')
        logging.info('===== Start to SLM predict =====')

        predict_dataset = InputDataset(args.predict_inputs, tokenizer)

        predict_dataloader = data.DataLoader(
            dataset=predict_dataset,
            shuffle=False,
            batch_size=args.predict_batch_size,
            num_workers=0,
            collate_fn=InputDataset.padding_collate
        )

        # Restore model from best checkpoint
        logging.info('Loading checkpoint %s...' % (args.init_checkpoint or args.save_path))
        checkpoint = torch.load(os.path.join(args.init_checkpoint or args.save_path, 'best-checkpoint'))
        step = checkpoint['global_step']
        slm.load_state_dict(checkpoint['model_state_dict'])

        logging.info('Global step of best-checkpoint: %s' % step)

        slm.eval()

        PREDICT_OUTPUT = f'{MODEL_PATH}/prediction.txt'

        fout_slm = open(PREDICT_OUTPUT, 'w')

        with torch.no_grad():
            for x_batch, seq_len_batch, uchars_batch, segments_batch, restore_orders in predict_dataloader:
                x_batch = x_batch.to(device)
                segments_batch_slm = slm(x_batch, seq_len_batch, mode='decode')
                for i in restore_orders:
                    uchars, segments_slm = uchars_batch[i], segments_batch_slm[i]
                    fout_slm.write(tokenizer.restore(uchars, segments_slm))

        fout_slm.close()

        # eval_command_slm = "bash run.sh valid %s" % ' '.join(args.save_path.split('/')[-1].rsplit('-'))
        eval_command_slm = f'perl {SCRIPT} {TRAINING_WORDS} {GOLD_TEST} {TEST_OUTPUT}'
        F_score = eval(eval_command_slm, 'slm', TEST_SCORE, True)

        if args.do_classifier or args.iterative_train:

            logging.info('===== Start to CLS predict =====')

            # Restore model from best checkpoint
            logging.info('Loading checkpoint %s...' % (args.init_checkpoint or args.save_path))
            checkpoint = torch.load(os.path.join(args.init_checkpoint or args.save_path, 'best-cls_checkpoint'))
            step = checkpoint['global_step']
            cls_model.load_state_dict(checkpoint['model_state_dict'])

            logging.info('Global step of best-checkpoint: %s' % step)

            cls_model.eval()

            CLS_PREDICT_OUTPUT = f'{MODEL_PATH}/prediction_cls.txt'
            cls_model.eval()
            fout_cls = open(CLS_PREDICT_OUTPUT, 'w')


            with torch.no_grad():
                for x_batch, seq_len_batch, uchars_batch, segments_batch, restore_orders in predict_dataloader:
                    segments_batch_cls = cls_model.generate_segments(x=x_batch.to(device), lengths=seq_len_batch)
                    for i in restore_orders:
                        uchars, segments_cls = uchars_batch[i], segments_batch_cls[i]
                        fout_cls.write(tokenizer.restore(uchars, segments_cls))

            fout_cls.close()
            # eval_command_cls = "bash run.sh valid_cls %s" % ' '.join(args.save_path.split('/')[-1].rsplit('-'))
            eval_command_cls = f'perl {SCRIPT} {TRAINING_WORDS} {GOLD_TEST} {CLS_TEST_OUTPUT}'
            F_score_cls = eval(eval_command_cls, 'CLS', CLS_TEST_SCORE, True)


def first_part(args):
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


    set_seed(args.seed)

    set_logger(args)
    logging.info(str(args))
    logging.info('\n======\n')

    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')

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

    if args.init_embedding_path:
        logging.info('Loading init embedding from %s...' % args.init_embedding_path)
        init_embedding = np.load(args.init_embedding_path)
    else:
        logging.info('Ramdomly Initializing character embedding...')
        init_embedding = None

    slm_config = model.SLMConfig.from_json_file(args.config_file)
    logging.info('Config Info:\n%s' % slm_config.to_json_string())

    slm = model.SegmentalLM(
        config=slm_config,
        init_embedding=init_embedding
    )
    logging.info('Model Info:\n%s' % slm)

    return device, tokenizer, slm_config, slm.to(device)


if __name__ == "__main__":
    start_time = time()

    main(parse_args())
    print(f'Process time: {time() - start_time }')

    
#     args = parse_args()
#     print(vars(args))
    
#     PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))

#     DATA = 'as'
#     DATA_PATH = f'{PROJECT_ROOT_PATH}/data/{DATA}'

#     # score script, words and gold.
#     SCRIPT = f'{PROJECT_ROOT_PATH}/data/score.pl'
#     TRAINING_WORDS = f'{DATA_PATH}/words.txt'
#     GOLD_TEST = f'{DATA_PATH}/test_gold.txt'

#     MODEL_DIR = args.save_path.split('/')[0]
#     MODE, DATA, MAX_SEG_LEN = args.save_path.split('/')[-1].rsplit('-')

#     MODEL_PATH = f'{PROJECT_ROOT_PATH}/{MODEL_DIR}/{MODE}-{DATA}-{MAX_SEG_LEN}'

#     # score and predcition output file.
#     # slm valid.
#     VALID_OUTPUT = f'{MODEL_PATH}/valid_prediction.txt'
#     VALID_SCORE = f'{MODEL_PATH}/valid_score.txt'

#     # slm prediction.
#     TEST_OUTPUT = f'{MODEL_PATH}/prediction.txt'
#     TEST_SCORE = f'{MODEL_PATH}/score.txt'

#     # cls valid.
#     CLS_VALID_OUTPUT = f'{MODEL_PATH}/valid_prediction_cls.txt'
#     CLS_VALID_SCORE = f'{MODEL_PATH}/valid_score_cls.txt'

#     # cls prediction.
#     CLS_TEST_OUTPUT = f'{MODEL_PATH}/prediction_cls.txt'
#     CLS_TEST_SCORE = f'{MODEL_PATH}/score_cls.txt'




#     eval_command_slm = f'perl {SCRIPT} {TRAINING_WORDS} {GOLD_TEST} {VALID_OUTPUT} > {VALID_SCORE}'
#     eval(eval_command_slm, 'slm_test')


# eval_command = eval_command_slm
# eval_command = f'perl {SCRIPT} --help > zzzz.txt'
# attr = 'zzzz_test'
# out = subprocess.Popen(eval_command.split(' '),
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.STDOUT)
# stdout, stderr = out.communicate()
# stdout = stdout.decode("utf-8")
# logging.info(f'{attr} Validation results:\n%s' % stdout)

# F_score = 0
# for line in stdout.split('\n'):
#     if line[:len('=== F MEASURE:')] == '=== F MEASURE:':
#         F_score = float(line.split('\t')[-1])

# print(f'{attr}, {F_score=}')

# os.popen(eval_command)


# print(f'{SCRIPT=}')
# print(f'{TRAINING_WORDS=}')


# print(f'{VALID_OUTPUT=}')
# print(f'{VALID_SCORE=}')