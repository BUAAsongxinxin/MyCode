# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-21 16:35
@Auth ： songxinxin
@File ：dataset_bilstm.py
"""
import os
import pickle
import json
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from config import *
from utils import *


class DataProcessor():
    def __init__(self, configs, logger):
        self.config = configs
        self.data_dir = configs.data_dir
        self.logger = logger
        self.max_seq_len = self.config.max_seq_len
        self.label2id = {'B-PER': 0,
                         'I-PER': 1,
                         'B-LOC': 2,
                         'I-LOC': 3,
                         'B-ORG': 4,
                         'I-ORG': 5,
                         'B-MISC': 6,
                         'I-MISC': 7,
                         'O': 8,
                         'START': 9,
                         'STOP': 10
                         }
        self.train_data, self.dev_data, self.test_data = self.init_data()
        self.word2id = self.build_vocab()
        self.train_ids = self.convert_to_idx(self.train_data)
        self.dev_ids = self.convert_to_idx(self.dev_data)
        self.test_ids = self.convert_to_idx(self.test_data)

    def init_data(self):
        self.logger.info('Initialing the data...')
        # train
        train_pickle_dir = os.path.join(self.data_dir + 'train_lstm.pkl')
        if os.path.exists(train_pickle_dir):
            with open(train_pickle_dir, 'rb') as f:
                train_data = pickle.load(f)
        else:
            train_data = self.load_data(os.path.join(self.data_dir + 'train.txt'))
            # train_dataset = self.pad_sequence(train_data)
            with open(train_pickle_dir, 'wb') as f:
                pickle.dump(train_data, f)
        self.logger.info('the lenght of train:{}'.format(len(train_data)))

        # dev
        dev_pickle_dir = os.path.join(self.data_dir + 'valid_lstm.pkl')
        if os.path.exists(dev_pickle_dir):
            with open(dev_pickle_dir, 'rb') as f:
                dev_data = pickle.load(f)
        else:
            dev_data = self.load_data(os.path.join(self.data_dir + 'valid.txt'))
            # dev_dataset = self.pad_sequence(dev_data)
            with open(dev_pickle_dir, 'wb') as f:
                pickle.dump(dev_data, f)
        self.logger.info('the lenght of dev:{}'.format(len(dev_data)))

        # test
        test_pickle_dir = os.path.join(self.data_dir + 'test_lstm.pkl')
        if os.path.exists(test_pickle_dir):
            with open(test_pickle_dir, 'rb') as f:
                test_data = pickle.load(f)
        else:
            test_data = self.load_data(os.path.join(self.data_dir + 'test.txt'))
            # test_dataset = self.pad_sequence(test_data)
            with open(test_pickle_dir, 'wb') as f:
                pickle.dump(test_data, f)
        self.logger.info('the lenght of test:{}'.format(len(test_data)))

        return train_data, dev_data, test_data

    def load_data(self, data_dir):
        self.logger.info(f'load data from {data_dir}')
        words = []
        labels = []
        pairs = []

        with open(data_dir) as f:
            lines = f.readlines()[:200]
            for (i, line) in enumerate(lines):
                line = line.strip()
                word_label = line.split('\t')
                word = word_label[0]
                label = word_label[-1]
                if len(words) == 0 and len(line) == 0:  # 排除.之后的连续空行，不然words[-1]会报错
                    continue
                elif (len(line) == 0 and words[-1] == '.') or i == len(lines) - 1:  # .结束一句或者文件最后一行
                    assert len(words) == len(labels)
                    pairs.append((words, labels))
                    # words_total.append(words)
                    # labels_total.append(labels)
                    # lengths_total.append(len(words))
                    words = []
                    labels = []
                elif len(line) == 0:  # 去除空行
                    continue
                else:
                    words.append(word)
                    labels.append(self.label2id[label])

        return pairs


    def build_vocab(self):
        """
        Create a dictionary of words, sorted by frequency
        :return:
        """
        vocab_dir = self.data_dir + 'vocab.txt'
        if os.path.exists(vocab_dir):
            with open(vocab_dir, 'r') as f:
                vocab_dict = json.load(f)
        else:
            word_dict = {}
            vocab_dict = {'<PAD>': 0, '<UNK>': 1}
            total_words = []
            for pairs in self.train_data:
                total_words.extend(pairs[0])
            for words in total_words:
                words = words.lower()
                if words not in word_dict:
                    word_dict[words] = 0
                word_dict[words] += 1

            # 建立词典
            sorted_dict = sorted(word_dict.items(), key=lambda x: x[1])  # 按value值进行排序
            for i, (k, v) in enumerate(sorted_dict):
                vocab_dict[k] = i + 2

            with open(vocab_dir, 'w') as f:
                json.dump(vocab_dict, f)

        return vocab_dict

    def convert_to_idx(self, data_str):
        pairs_id = []
        for words, tags in data_str:
            words_id = []
            for word in words:
                if word in self.word2id.keys():
                    words_id.append(self.word2id[word])
                else:
                    words_id.append(self.word2id['<UNK>'])
            pairs_id.append((words_id,tags))
        return pairs_id

    # def pad_sequence(self, x):
    #     '''
    #     pad squence and convert vocab to idx dataset
    #     :param x: data
    #     :return:
    #     '''
    #     words_total = x['words_total']
    #     labels_total = x['labels_total']
    #     lengths_total = x['lengths_total']
    #
    #     words_paded = []
    #     labels_paded = []
    #     lengths_paded = []
    #     assert len(words_total) == len(labels_total) == len(lengths_total)
    #     for words, labels, length in zip(words_total, labels_total, lengths_total):
    #         word_idx = []
    #         for word in words:
    #             if word in self.word2id.keys():
    #                 word_idx.append(self.word2id[word])
    #             else:
    #                 word_idx.append(self.word2id['<UNK>'])
    #         words = word_idx
    #         assert len(words) == len(labels) == length
    #         # if length >= self.max_seq_len:
    #         #     length = self.max_seq_len
    #         #     words = words[:length]
    #         #     labels = labels[:length]
    #         # elif length < self.max_seq_len:
    #         #     pad_num = self.max_seq_len - length
    #         #     words = words + [0] * pad_num
    #         #     labels = labels + [9] * pad_num  # pad
    #         # else:
    #         #     pass
    #
    #         words_paded.append(words)
    #         labels_paded.append(labels)
    #         lengths_paded.append(length)
    #
    #     return TensorDataset(torch.tensor(words_paded), torch.tensor(labels_paded), torch.tensor(lengths_paded))

    # def __getitem__(self, item):
    #     return self.train_dataset[item], self.dev_dataset[item], self.test_dataset[item]
    #
    # def __len__(self):
    #     return len(self.train_dataset), len(self.dev_dataset), len(self.test_dataset)

    # def get_dataloader(self):
    #     train_loader = DataLoader(self.train_data,
    #                               batch_size=self.config.batch_size,
    #                               shuffle=True)
    #
    #     dev_loader = DataLoader(self.dev_data,
    #                               batch_size=self.config.batch_size,
    #                               shuffle=False)
    #
    #     test_loader = DataLoader(self.test_data,
    #                               batch_size=self.config.batch_size,
    #                               shuffle=False)
    #     return train_loader, dev_loader, test_loader


if __name__ == '__main__':
    config = config()
    log = log('../log/log.txt')
    data_processor = DataProcessor(config, log)
    print(data_processor.train_data)
