# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-21 16:35
@Auth ： songxinxin
@File ：dataset_bert.py
"""
import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from config import *
from utils import *


class ner_bert_dataset(Dataset):
    def __init__(self, configs, logger):
        self.config = configs
        self.data_dir = configs.data_dir
        self.logger = logger
        self.label2id = {'<PAD>': 0, # 0留给pad
                         'B-PER': 1,
                         'I-PER': 2,
                         'B-LOC': 3,
                         'I-LOC': 4,
                         'B-ORG': 5,
                         'I-ORG': 6,
                         'B-MISC': 7,
                         'I-MISC': 8,
                         'O': 9,
                         '[CLS]': 10,
                         }  # [SEP]不知道要不要加，先不加了
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.train_data, self.dev_data, self.test_data = self.init_data()

    def init_data(self):
        self.logger.info('Initialing the data...')
        # train
        train_pkl_dir = os.path.join(self.data_dir + 'train.pkl')
        if os.path.exists(train_pkl_dir):
            with open(train_pkl_dir, 'rb') as f:
                train_data = pickle.load(f)
        else:
            train_data = self.load_data(os.path.join(self.data_dir + 'train.txt'))
            with open(train_pkl_dir, 'wb') as f:
                pickle.dump(train_data, f)
        self.logger.info('the lenght of train:{}'.format(len(train_data)))

        # dev
        dev_pkl_dir = os.path.join(self.data_dir + 'valid.pkl')
        if os.path.exists(dev_pkl_dir):
            with open(dev_pkl_dir, 'rb') as f:
                dev_data = pickle.load(f)
        else:
            dev_data = self.load_data(os.path.join(self.data_dir + 'valid.txt'))
            with open(dev_pkl_dir, 'wb') as f:
                pickle.dump(dev_data, f)
        self.logger.info('the lenght of dev:{}'.format(len(dev_data)))

        # test
        test_pkl_dir = os.path.join(self.data_dir + 'test.pkl')
        if os.path.exists(test_pkl_dir):
            with open(test_pkl_dir, 'rb') as f:
                test_data = pickle.load(f)
        else:
            test_data = self.load_data(os.path.join(self.data_dir + 'test.txt'))
            with open(test_pkl_dir, 'wb') as f:
                pickle.dump(test_data, f)
        self.logger.info('the lenght of test:{}'.format(len(test_data)))

        return train_data, dev_data, test_data

    def load_data(self, data_dir):
        self.logger.info(f'load data from {data_dir}')
        input_ids = []
        token_type_ids = []
        attention_mask = []
        labels_bert = []
        lengths = []

        words = []
        labels = []

        with open(data_dir) as f:
            lines = f.readlines()[:200]
            for (i, line) in enumerate(lines):
                line = line.strip()
                word_label = line.split('\t')
                word = word_label[0]
                label = word_label[-1]
                if len(words) == 0 and len(line) == 0:  # 排除.之后的连续空行，不然words[-1]会报错
                    continue
                elif (len(line) == 0 and words[-1] == '.') or i == len(lines) - 1 or len(words) == 500:  # .结束一句或者文件最后一行或者超过510,后续会加上[CLS]和[SEP]
                    assert len(words) == len(labels)
                    # input_idx = [101] + self.tokenizer.convert_tokens_to_ids(words) + [102]  # 因为这个是一个词一个词性，若用字符串然后tokenize的话会导致label和词对不上
                    input_idx = [101] + self.tokenizer.convert_tokens_to_ids(words)
                    assert len(input_idx) == len(words) + 1
                    token_type_idx = torch.tensor([0] * len(input_idx))
                    attention_mask_idx = torch.tensor([1] * len(input_idx))

                    input_ids.append(torch.tensor(input_idx))  # 使用pad_squence时，元素要是tensor列表
                    token_type_ids.append(token_type_idx)
                    attention_mask.append(attention_mask_idx)
                    labels_bert.append(torch.tensor([10] + labels))
                    lengths.append(len(labels))  # 记录长度

                    words = []
                    labels = []
                elif len(line) == 0:  # 去除空行
                    continue
                else:
                    words.append(word)
                    labels.append(self.label2id[label])

        input_ids = pad_sequence(input_ids, batch_first=True)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        labels = pad_sequence(labels_bert, batch_first=True)
        lengths = torch.tensor(lengths)

        return TensorDataset(input_ids, token_type_ids, attention_mask, labels, lengths)

        # # 输出labels的类别
        # labels = []
        # for seq_labels in seq_out:
        #     labels = labels + seq_labels
        # print(set(labels))
    def get_dataloader(self):
        train_loader = DataLoader(self.train_data,
                                  batch_size=self.config.batch_size,
                                  shuffle=True)

        dev_loader = DataLoader(self.dev_data,
                                  batch_size=self.config.batch_size,
                                  shuffle=False)

        test_loader = DataLoader(self.test_data,
                                  batch_size=self.config.batch_size,
                                  shuffle=False)
        return train_loader, dev_loader, test_loader

    def __getitem__(self, item):
        return self.train_data[item], self.dev_data[item], self.test_data[item]

    def __len__(self):
        return len(self.train_data), len(self.dev_data), len(self.test_data)


if __name__ == '__main__':
    config = config()
    log = log('../log/log.txt')
    dataset = ner_bert_dataset(config, log)
    train_loader = dataset.get_dataloader()
    print(train_loader)
    print(dataset.train_data[0])
