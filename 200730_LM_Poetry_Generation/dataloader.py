# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-30 16:43
@Auth ： songxinxin
@File ：dataloader.py
"""
import os
import pandas as pd
from torchtext import data


# 处理data方便torchtext处理
def load_data(data_path):
    file_path = data_path + 'poetryFromTang.txt'
    target_path = data_path + 'train.csv'
    if not os.path.exists(target_path):
        with open(file_path, 'r') as f:
            lines = f.read().split('\n\n')  # 记得去掉
        lines = list(map(lambda x: x.replace('\n', ''), lines))  # 转化成一行
        df = pd.DataFrame()
        df['sent'] = lines
        df.to_csv(target_path, index=False, encoding='utf_8_sig')  # 写入csv要使用utf_8_sig，不然会乱码

    return target_path


def dataset2dataloader(data_path, batch_size=16, device='cpu'):
    target_path = load_data(data_path)
    # 创建Field对象
    SENT = data.Field(sequential=True, tokenize=tokenizer, lower=False, init_token='<START>', eos_token='<END>', batch_first=True)
    # 构建dataset
    train_dataset, _ = data.TabularDataset.splits(path='', train=target_path, validation=target_path, format='csv',
                                                  skip_header=True, fields=[('sent', SENT)])
    # 生成词表
    SENT.build_vocab(train_dataset,vectors="glove.6B.100d")
    # 构造迭代器
    train_iter = data.BucketIterator(train_dataset, batch_size=batch_size, sort_key=lambda x:len(x.sent), shuffle=False, device=device)

    print('Data loading is finished!')
    print(f'The length of data is {len(train_dataset)}')
    return train_iter, SENT.vocab


def tokenizer(text):
    return list(text)


if __name__ == '__main__':
    train_iter, vocab = dataset2dataloader('./')
    for batch in train_iter:
        print(batch.sent)
        print(batch.sent.shape)
        break
