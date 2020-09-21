# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-21 22:21
@Auth ： songxinxin
@File ：train.py
"""
import torch.nn as nn
import torch
import time
from torch.optim import SGD
from bilstm_crf.model import BiLSTM_CRF
from bilstm_crf.dataset_bilstm import DataProcessor
from utils import log, calculate_score


class BiLSTMCRFTrainer:
    def __init__(self, config):
        self.config = config
        self.model_dir = config.model_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger = log(config.log_file)
        # data
        self.dataset = DataProcessor(config, self.logger)
        # model
        self.model = BiLSTM_CRF(config, len(self.dataset.word2id), self.dataset.label2id, self.device).to(self.device)
        self.optimizer = SGD(self.model.parameters(), lr=config.lr)

        self.train_data = self.dataset.train_ids
        self.dev_data = self.dataset.dev_ids
        self.test_data = self.dataset.test_ids

    def train(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f'total paras is {total_params}')
        self.logger.info(f'trainable params is {trainable_params}')

        self.logger.info('----------START TRAIN----------')
        # self.load_model()
        best_val_F1 = 0.0
        for epoch in range(self.config.epoch):
            start_time = time.time()
            train_loss, train_precison, train_recall, train_F1 = self.train_epoch()
            val_loss, val_precison, val_recall, val_F1 = self.eval('dev')

            if val_F1 > best_val_F1:
                best_val_F1 = val_F1
                self.logger.info(f'save model from epoch_{epoch}')
                self.save_model()

            test_loss, test_precison, test_recall, test_F1= self.eval('test')
            end_time = time.time()
            epoch_time = end_time - start_time

            self.logger.info('epoch:[{}/{}] time: {:.4f}'.format(epoch, self.config.epoch, epoch_time))
            self.logger.info('Train: loss:{:.4f}, precison:{:.4f}, recall:{:.4f}, F1_score:{:.4f}' \
                            .format(train_loss, train_precison, train_recall, train_F1))

            self.logger.info('Dev: loss:{:.4f}, precison:{:.4f}, recall:{:.4f}, F1_score:{:.4f}' \
                             .format(val_loss, val_precison, val_recall, val_F1))

            self.logger.info('Test: loss:{:.4f}, precison:{:.4f}, recall:{:.4f}, F1_score:{:.4f}' \
                             .format(test_loss, test_precison, test_recall, test_F1))
            self.logger.info('-------------------\n')

        self.test()

    def train_epoch(self):
        self.model.train()
        train_loss = 0.0
        true_labels = []
        pred_labels = []

        for sentence, tags in self.train_data:
            self.optimizer.zero_grad()

            sentence_in = prepare_sequence(sentence).to(self.device)
            targets = torch.tensor(tags, dtype=torch.long, device=self.device).to(self.device)

            loss = self.model.neg_log_likelihood(sentence_in, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

            _, pred = self.model(sentence_in)
            train_loss += loss.item()
            true_labels.extend(tags)
            pred_labels.extend(pred)

        precision, recall, F1_score = calculate_score(true_labels, pred_labels, len(self.dataset.label2id))
        # precision, recall, F1_score = calculate_score(torch.tensor(true_labels), torch.tensor(pred_labels), len(self.dataset.label2id))
        return train_loss / len(self.train_data), precision, recall, F1_score

    def eval(self, mode):
        self.model.eval()
        if mode == 'dev':
            data_eval = self.dev_data
        else:
            data_eval = self.test_data

        total_loss = 0.0
        true_labels = []
        pred_labels = []

        with torch.no_grad():
            for sentence, tags in data_eval:
                sentence_in = prepare_sequence(sentence).to(self.device)
                targets = torch.tensor(tags, dtype=torch.long).to(self.device)

                loss = self.model.neg_log_likelihood(sentence_in, targets)
                _, pred = self.model(sentence_in)
                total_loss += loss.item()

                true_labels.extend(tags)
                pred_labels.extend(pred)

        precision, recall, F1_score = calculate_score(true_labels, pred_labels,  len(self.dataset.label2id))
        return total_loss / len(data_eval), precision, recall, F1_score

    def test(self):
        self.load_model()
        test_loss, test_precision, test_recall, test_F1 = self.eval('test')
        self.logger.info('test_loss: {:.4f}, test_precision:{:.4f} test_recall: {:.4f}, test_F1:{:.4f}'.\
                         format(test_loss, test_precision, test_recall, test_F1))

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_dir)

    def load_model(self):
        self.logger.info(f'loading model from {self.model_dir}')
        self.model.load_state_dict(torch.load(self.model_dir))

def prepare_sequence(seq_ids):
    # convert list to tensor
    return torch.tensor(seq_ids, dtype=torch.long)



