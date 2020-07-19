# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-07 15:19
@Auth ： songxinxin
@File ：config.py
"""
import argparse


def config():
    parser = argparse.ArgumentParser(description='config of snli')
    parser.add_argument('--data_path', type=str, default='./snli_1.0/')
    parser.add_argument('--train_dir', type=str, default='./snli_1.0/snli_1.0_train.jsonl')
    parser.add_argument('--dev_dir', type=str, default='./snli_1.0/snli_1.0_dev.jsonl')
    parser.add_argument('--test_dir', type=str, default='./snli_1.0/snli_1.0_test.jsonl')
    parser.add_argument('--model_dir', type=str, default='./model/best.model')
    parser.add_argument('--model', type=str, default='esim', help='[bert, ESIM]')
    parser.add_argument('--log_file', type=str, default='./logging/log.txt')

    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--max_premise_len', type=int, default=30)
    parser.add_argument('--max_hypothesis_len', type=int, default=40)  # 具体跑一下看一下情况

    # lstm
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=600)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--embedding_file', type=str, default='../glove/glove.6B.300d.txt')  # 试试embedding

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout_emb', type=float, default=0.)
    parser.add_argument('--dropout_fc', type=float, default=0.3)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--train_bert', type=bool, default=False)

    return parser.parse_args()
