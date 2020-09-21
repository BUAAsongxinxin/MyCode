# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-21 22:21
@Auth ： songxinxin
@File ：train_bert.py
"""
import torch.nn as nn
import torch
import time
from torch.optim import Adam
from bert.dataset_bert import *
from bert.bert import BertNer
from utils import log


class BertTrainer:
    def __init__(self, config):
        self.config = config
        self.model_dir = config.model_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger = log(config.log_file)
        # model
        self.model = BertNer(config).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=config.lr)
        self.loss_func = nn.CrossEntropyLoss()  # softmax已经在fc的时候计算过了
        self.dataset = ner_bert_dataset(config, self.logger)
        self.train_loader, self.dev_loader, self.test_loader = self.dataset.get_dataloader()

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
        total_num = 0
        true_labels = []
        pred_labels = []

        for i, (input_ids, token_type_ids, attention_mask, labels, length) in enumerate(self.train_loader):  # lables要取一下真实长度
            total_num += input_ids.shape[0]

            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(input_ids, token_type_ids, attention_mask)  # [batch_size, seq_len, 9]

            outputs_ = outputs.reshape(-1, outputs.shape[-1])
            labels_ = labels.reshape(-1)  # 改成[N,C]的格式
            loss = self.loss_func(outputs_, labels_)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

            _, preds = torch.max(outputs, dim=-1)
            batch_true_labels = []
            batch_pred_labels = []
            for i in range(len(length)):  # 一共有几条数据
                batch_true_labels.extend(labels[i, 1:length[i]].cpu().numpy().tolist())  # # 0是[CLS]
                batch_pred_labels.extend(preds[i, 1:length[i]].cpu().numpy().tolist())

            true_labels.extend(batch_true_labels)
            pred_labels.extend(batch_pred_labels)

            train_loss += loss.item()

        precision, recall, F1_score = self.calculate_score(torch.tensor(true_labels), torch.tensor(pred_labels))
        train_loss = train_loss / total_num

        return train_loss, precision, recall, F1_score

    def eval(self, mode):
        self.model.eval()
        if mode == 'dev':
            data_loader = self.dev_loader
        else:
            data_loader = self.test_loader

        total_loss = 0.0
        total_num = 0
        true_labels = []
        pred_labels = []

        with torch.no_grad():
            for i, (input_ids, token_type_ids, attention_mask, labels, length) in enumerate(data_loader):
                total_num += input_ids.shape[0]

                input_ids = input_ids.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(input_ids, token_type_ids, attention_mask)  # [batch_size, seq_len, 9]

                outputs_ = outputs.reshape(-1, outputs.shape[-1])
                labels_ = labels.reshape(-1)  # 改成[N,C]的格式
                loss = self.loss_func(outputs_, labels_)

                _, preds = torch.max(outputs, dim=-1)
                batch_true_labels = []
                batch_pred_labels = []
                for i in range(len(length)):  # 一共有几条数据
                    batch_true_labels.extend(labels[i, 1:length[i]].cpu().numpy().tolist())  # 0是[CLS]
                    batch_pred_labels.extend(preds[i, 1:length[i]].cpu().numpy().tolist())

                true_labels.extend(batch_true_labels)
                pred_labels.extend(batch_pred_labels)

                total_loss += loss.item()

            loss_val = total_loss / total_num
            precision, recall, F1_score = self.calculate_score(torch.tensor(true_labels), torch.tensor(pred_labels))

        return loss_val, precision, recall, F1_score

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

    def calculate_score(self, true, pred):
        true = true.view(-1)
        pred = pred.view(-1)  # 拉成直的
        confusion_matrix = torch.zeros(self.config.num_classes, self.config.num_classes)
        for t, p in zip(true, pred):
            confusion_matrix[t, p] +=1

        precison = torch.diag(confusion_matrix) / torch.sum(confusion_matrix + 1e-5, dim=1)  # [num_classes, 1]
        recall = torch.diag(confusion_matrix) / torch.sum(confusion_matrix + 1e-5, dim=0)

        F1 = (precison * recall * 2) / (precison + recall + 1e-5)

        return torch.mean(precison), torch.mean(recall), torch.mean(F1)

