# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-10 10:46
@Auth ： songxinxin
@File ：dataset_esim.py
"""
import json
import os
import collections

import torch
from torch.utils.data import DataLoader, TensorDataset
from config import *
import re
import random


def lowerPunctuation(text):
    punctuation = '!,.;:?"\''
    text = re.sub(r'[{}]+'.format(punctuation), '', text)

    return text.strip().lower()


def add_embedding(embedding_matrix, words, vector_size):
    # 将<unk>,<pad>加入,随机初始化值即可
    for word in words:
        vector = torch.randn(1, vector_size).numpy().tolist()
        embedding_matrix[word] = vector[0]

    return embedding_matrix


class SNLIDatasetESIM:
    def __init__(self, configs, logger):
        super(SNLIDatasetESIM, self).__init__()
        self.config = configs
        self.logger = logger
        self.train_dir = configs.train_dir
        self.dev_dir = configs.dev_dir
        self.test_dir = configs.test_dir
        self.embed_file = configs.embedding_file
        self.label2dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        # self.max_premise_len = configs.max_premise_len
        # self.max_hypothesis_len = configs.max_hypothesis_len

        self.train_data, self.dev_data, self.test_data = self.init_data()
        self.word2dict = self.build_vocab()
        self.vocab_size = len(self.word2dict)
        self.embedding = self.make_embedding()

        self.max_premise_len = max(self.train_data['lengths_promise'])
        self.max_hypothesis_len = max(self.train_data['lenghts_hypotheses'])

    def init_data(self):
        self.logger.info('Initializing train data ...')
        train_pkl_dir = self.config.data_path + 'train_esim.json'
        if os.path.exists(train_pkl_dir):
            self.logger.info('Found train_esim.json')
            with open(train_pkl_dir, 'r') as f:
                train_data = json.load(f)
        else:
            train_data = self.load_data(self.train_dir)
            with open(train_pkl_dir, 'w') as f:
                json.dump(train_data, f)

        self.logger.info('The length of train: {}'.format(len(train_data['labels'])))

        self.logger.info('Initializing dev data ...')
        dev_pkl_dir = self.config.data_path + 'dev_esim.json'
        if os.path.exists(dev_pkl_dir):
            self.logger.info('Found dev_esim.json')
            with open(dev_pkl_dir, 'r') as f:
                dev_data = json.load(f)
        else:
            dev_data = self.load_data(self.dev_dir)
            with open(dev_pkl_dir, 'w') as f:
                json.dump(dev_data, f)
        self.logger.info('The length of dev: {}'.format(len(dev_data['labels'])))

        self.logger.info('Initializing test data ...')
        test_pkl_dir = self.config.data_path + 'test_esim.json'
        if os.path.exists(test_pkl_dir):
            self.logger.info('Found test_esim.json')
            with open(test_pkl_dir, 'r') as f:
                test_data = json.load(f)
        else:
            test_data = self.load_data(self.test_dir)
            with open(test_pkl_dir, 'w') as f:
                json.dump(test_data, f)
        self.logger.info('The length of test: {}'.format(len(test_data['labels'])))

        return train_data, dev_data, test_data

    def load_data(self, path):
        self.logger.info(f'Load Data from {path}')
        labels = []
        premises = []
        hypotheses = []
        lenghts_promise = []  # 记录每个的长度
        lenghts_hypotheses = []
        with open(path, 'r') as f:
            lines = f.readlines()
            if len(lines) > 100000:
                lines = random.sample(lines, 20000)
            else:
                lines = random.sample(lines, 1000)
            for line in lines:
                line = json.loads(line)
                if line['gold_label'] not in self.label2dict:  # 去除不相关的label
                    continue
                label = self.label2dict[line['gold_label']]
                premise = line['sentence1']
                hypothesis = line['sentence2']
                premise = lowerPunctuation(premise)
                hypothesis = lowerPunctuation(hypothesis)

                labels.append(label)
                premises.append(premise.split(' '))
                hypotheses.append(hypothesis.split(' '))
                lenghts_promise.append(len(premise))
                lenghts_hypotheses.append(len(hypothesis))

        data_dict = {"labels": labels, "premises": premises, "hypotheses": hypotheses,
                     "lengths_promise": lenghts_promise, "lenghts_hypotheses": lenghts_hypotheses}
        return data_dict

    def build_vocab(self):
        vocab_dir = self.config.data_path + 'vocab.txt'
        vocab_dict = {}
        if os.path.exists(vocab_dir):
            with open(vocab_dir, 'r') as f:
                vocab_dict = json.load(f)
        else:
            max_len1 = 0
            max_len2 = 0
            # train_data = self.load_data(self.train_dir)
            train_data = self.train_data
            words = []
            for premise in train_data['premises']:
                if max_len1 < len(premise):
                    max_len1 = len(premise)
                words.extend(premise)
            for hypothesis in train_data['hypotheses']:
                if max_len2 < len(hypothesis):
                    max_len2 = len(hypothesis)
                words.extend(hypothesis)
            self.logger.info(f'max_len_premise:{max_len1}, max_len_hyphotheses:{max_len2}')

            self.logger.info(f'num of words :{len(words)}')
            words_count = collections.Counter(words)
            # print(words_count)
            self.logger.info(f'lenght of vocab:{len(words_count)}')

            vocab_dict['<PAD>'] = 0
            vocab_dict['<UNK>'] = 1
            vocab_dict['<BOS>'] = 2
            vocab_dict['<EOS>'] = 3

            for i, word in enumerate(words_count.most_common(len(words_count))):  # word（词，频数）
                vocab_dict[word[0]] = i + 4

            with open(vocab_dir, 'w') as f:
                json.dump(vocab_dict, f)

        return vocab_dict

    def make_embedding(self):
        self.logger.info('getting embedding')
        embedding_matrix = {}
        with open(self.embed_file, 'r')as f:
            lines = f.readlines()
            vector_size = len(lines[0].strip().split(' ')) - 1
            for line in lines:
                word = line.strip().split(' ')[0]
                vec = line.strip().split(' ')[1:]
                vec_num = [float(i) for i in vec]
                assert len(vec_num) == vector_size
                embedding_matrix[word] = vec_num

            embedding_matrix = add_embedding(embedding_matrix, ['<UNK>', '<PAD>', '<BOS>', '<EOS>'], vector_size)
        f.close()

        embedding = []
        for word, index in self.word2dict.items():
            if word in embedding_matrix.keys():
                embedding.append(embedding_matrix[word])
            else:
                embedding.append(embedding_matrix['<UNK>'])

        return torch.tensor(embedding)

    def get_dataloaders(self):
        train_dataset = self.pad_sequence(self.train_data)
        dev_dataset = self.pad_sequence(self.dev_data)
        test_dataset = self.pad_sequence(self.test_data)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config.batch_size,
                                  shuffle=True)

        dev_loader = DataLoader(dev_dataset,
                                batch_size=self.config.batch_size,
                                shuffle=False)

        test_loader = DataLoader(test_dataset,
                                 batch_size=self.config.batch_size,
                                 shuffle=False)
        return train_loader, dev_loader, test_loader

    def word2id(self, sentence):
        sent_id = []
        for word in sentence:
            if word in self.word2dict:
                sent_id.append(self.word2dict[word])
            else:
                sent_id.append(1)  # <UNK>

        return sent_id

    def pad_sequence(self, data):
        premises = []
        hypotheses = []
        labels = torch.tensor(data['labels'])
        # 为了防止test时，test的length更大导致mask出现问题
        for i, len_pre in enumerate(data['lengths_promise']):
            if len_pre > self.max_premise_len:
                data['lengths_promise'][i] = self.max_premise_len
        lengths_promise = torch.tensor(data['lengths_promise'])

        for i, len_hypo in enumerate(data['lenghts_hypotheses']):
            if len_hypo > self.max_hypothesis_len:
                data['lenghts_hypotheses'][i] = self.max_hypothesis_len
        lenghts_hypotheses = torch.tensor(data['lenghts_hypotheses'])

        for i in range(len(labels)):
            premise = ['<BOS>'] + data['premises'][i] + ['<EOS>']
            hypothesis = ['<BOS>'] + data['hypotheses'][i] + ['<EOS>']

            if len(premise) > self.max_premise_len:
                premise = premise[:self.max_premise_len]
            elif len(premise) < self.max_premise_len:
                premise = premise + ['<PAD>']*(self.max_premise_len - len(premise))
            else:
                pass

            assert len(premise) == self.max_premise_len

            if len(hypothesis) > self.max_hypothesis_len:
                hypothesis = hypothesis[:self.max_hypothesis_len]
            elif len(hypothesis) < self.max_hypothesis_len:
                hypothesis = hypothesis + ['<PAD>'] * (self.max_hypothesis_len - len(hypothesis))
            else:
                pass

            assert len(hypothesis) == self.max_hypothesis_len

            premises.append(self.word2id(premise))
            hypotheses.append(self.word2id(hypothesis))

        premises = torch.tensor(premises)
        hypotheses = torch.tensor(hypotheses)

        return TensorDataset(premises, hypotheses, labels, lengths_promise, lenghts_hypotheses)  # 一定都是tensor


if __name__ == '__main__':
    config = config()
    snli_dataset = SNLIDatasetESIM(config)
    train_loader, dev_loader, test_loader = snli_dataset.get_dataloaders()
    print(train_loader)
