# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-21 16:41
@Auth ： songxinxin
@File ：config.py
"""
import argparse


def config():
    parser = argparse.ArgumentParser()

    # data
    data_arg = parser.add_argument_group('Data')
    data_arg.add_argument('--data_dir', type=str, default='./ner/')

    # model
    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--model', type=str, default='bilstm_crf', help='[bert, bilstm_crf]')
    model_arg.add_argument('--batch_size', type=int, default=32)
    model_arg.add_argument('--lr', type=float, default=1e-4)
    model_arg.add_argument('--hidden_dim', type=int, default=300)
    model_arg.add_argument('--num_classes', type=int, default=11)
    model_arg.add_argument('--dropout_bert', type=float, default=0.3)
    model_arg.add_argument('--test', type=bool, default=False)
    model_arg.add_argument('--epoch', type=int, default=15)

    # bilstm-crf
    model_arg.add_argument('--max_seq_len', type=int, default=200)
    model_arg.add_argument('--embedding_dim', type=int, default=300)


    # other
    other_arg = parser.add_argument_group('log')
    other_arg.add_argument('--log_file', type=str, default='./log/log.txt')
    other_arg.add_argument('--model_dir', type=str, default='./model/best.model')


    return parser.parse_args()
