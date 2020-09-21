# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-30 16:29
@Auth ： songxinxin
@File ：config.py
"""
import argparse

def praser():
    praser = argparse.ArgumentParser()

    # data
    praser.add_argument('--data_dir', type=str, default='./')
    praser.add_argument('--batch_size', type=int, default=32)

    # model
    praser.add_argument('--embedding_dim', type=int, default=300)
    praser.add_argument('--hidden_dim', type=int, default=300)
    praser.add_argument('--num_layers', type=int, default=2)
    praser.add_argument('--lr', type=float, default=1e-3)
    praser.add_argument('--max_len', type=int, default=10)  # generate时的最大长度
    praser.add_argument('--epoch_num', type=int, default=20)

    # other
    praser.add_argument('--log_dir', type=str, default='log.txt')
    praser.add_argument('--test', type=bool, default=False)

    return praser.parse_args()