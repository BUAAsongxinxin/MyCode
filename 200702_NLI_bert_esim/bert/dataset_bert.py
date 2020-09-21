# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-07 15:19
@Auth ： songxinxin
@File ：dataset_bert.py
"""
import os
import pickle
import torch
import json
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from config import *


class SNLIDataSet(Dataset):
    def __init__(self, configs):
        self.config = configs
        self.train_dir = configs.train_dir
        self.dev_dir = configs.dev_dir
        self.test_dir = configs.test_dir
        self.label2dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.train_data, self.dev_data, self.test_data = self.init_data()

    def init_data(self):
        print('Initializing train data ...')
        train_pkl_dir = self.config.data_path + 'train.pkl'
        if os.path.exists(train_pkl_dir):
            print('Found train.pkl')
            with open(train_pkl_dir, 'rb') as f:
                train_data = pickle.load(f)
        else:
            train_data = self.load_data(self.train_dir)
            with open(train_pkl_dir, 'wb') as f:
                pickle.dump(train_data, f)
        print('The length of train: {}'.format(len(train_data)))

        print('Initializing dev data ...')
        dev_pkl_dir = self.config.data_path + 'dev.pkl'
        if os.path.exists(dev_pkl_dir):
            print('Found dev.pkl')
            with open(dev_pkl_dir, 'rb') as f:
                dev_data = pickle.load(f)
        else:
            dev_data = self.load_data(self.dev_dir)
            with open(dev_pkl_dir, 'wb') as f:
                pickle.dump(dev_data, f)
        print('The length of dev: {}'.format(len(dev_data)))

        print('Initializing test data ...')
        test_pkl_dir = self.config.data_path + 'test.pkl'
        if os.path.exists(test_pkl_dir):
            print('Found test.pkl')
            with open(test_pkl_dir, 'rb') as f:
                test_data = pickle.load(f)
        else:
            test_data = self.load_data(self.test_dir)
            with open(test_pkl_dir, 'wb') as f:
                pickle.dump(test_data, f)
        print('The length of test: {}'.format(len(test_data)))

        return train_data, dev_data, test_data

    def load_data(self, path):
        print('Load data from {}'.format(path))
        token_ids = []
        seg_ids = []
        mask_ids = []
        labels = []
        with open(path, 'r') as f:
            lines = f.readlines()[:90]
            for line in lines:
                line = json.loads(line)
                if line['gold_label'] not in self.label2dict:
                    # print(line['gold_label'])
                    continue
                labels.append(self.label2dict[line['gold_label']])
                premise = line['sentence1']
                hypothesis = line['sentence2']
                premise_idx = self.tokenizer.encode(premise)
                hypothesis_idx = self.tokenizer.encode(hypothesis)
                pair_token_idx = [101] + premise_idx + [102] + hypothesis_idx + [102]  # [101]-[CLS] [102]-[SEP]
                seg_idx = torch.tensor([0] * (len(premise_idx) + 2) + [1] * (len(hypothesis_idx) + 1))  # sentence1,2
                mask_idx = torch.tensor([1] * (len(pair_token_idx)))  # mask padded values

                token_ids.append(torch.tensor(pair_token_idx))
                seg_ids.append(seg_idx)
                mask_ids.append(mask_idx)

        token_ids = pad_sequence(token_ids, batch_first=True)  # input为list类型，元素为tensor
        seg_ids = pad_sequence(seg_ids, batch_first=True)
        mask_ids = pad_sequence(mask_ids, batch_first=True)
        labels = torch.tensor(labels)

        return TensorDataset(token_ids, seg_ids, mask_ids, labels)

    def get_dataloaders(self):
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

    def __len__(self):
        return len(self.train_data), len(self.dev_data), len(self.test_data)

    def __getitem__(self, item):
        return self.train_data[item], self.dev_data[item], self.test_data[item]


if __name__ == '__main__':
    config = config()
    snli_dataset = SNLIDataSet(config)
