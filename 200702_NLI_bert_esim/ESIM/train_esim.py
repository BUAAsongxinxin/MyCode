# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-10 22:49
@Auth ： songxinxin
@File ：train_esim.py
"""
from ESIM.dataset_esim import *
from ESIM.esim import *
from torch.optim import Adam
import torch
import time
from ESIM.utils import log

class TrainerEsim:
    def __init__(self, config):
        super(TrainerEsim, self).__init__()
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger = log(config.log_file)

        self.snli_dataset = SNLIDatasetESIM(config, self.logger)
        self.train_loader, self.dev_loader, self.test_loader = self.snli_dataset.get_dataloaders()
        self.max_premise_len = self.snli_dataset.max_premise_len
        self.max_hypothesis_len = self.snli_dataset.max_hypothesis_len
        self.model = Esim(config, self.snli_dataset.vocab_size, self.snli_dataset.embedding).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=config.lr)

        self.loss_func = nn.CrossEntropyLoss()

    def train(self):
        total_para = sum(p.numel() for p in self.model.parameters())
        self.logger.info("total parameters:{}".format(total_para))
        # print("total parameters:{}".format(total_para))
        trainable_para = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info("trainable parameters:{}".format(trainable_para))
        # print("trainable parameters:{}".format(trainable_para))

        if os.path.exists(self.config.model_dir):
            self.logger.info('loading the existed model……')
            self.model = self.load_model()
            _, val_acc = self.evaluate('dev')
            self.logger.info('load the best model success!')
        else:
            val_acc = 0.0
            self.logger.info('No existed model')

        self.logger.info('--------------- Start Train ---------------')
        # print('--------------- Start Train ---------------')
        best_val_acc = val_acc
        for cur_epoch in range(self.config.epoch):
            start_time = time.time()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate('dev')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(cur_epoch)
            test_loss, test_acc = self.evaluate('test')

            end_time = time.time()
            use_time = end_time - start_time

            self.logger.info('epoch: {}/{} | train_loss: {:.3f}, train_acc: {:.3f} | val_loss: {:.3f}, val_acc: {:.3f} '
                  '|test_loss: {:.3f}, test_acc: {:.3f} | time: {:.3f}'.\
                  format(cur_epoch, self.config.epoch, train_loss, train_acc,
                         val_loss, val_acc, test_loss, test_acc, use_time))

        self.logger.info('Train Finished!')

    def train_epoch(self):
        self.model.train()
        train_loss = 0.0
        correct_num = 0

        count_train = len(self.snli_dataset.train_data['labels'])
        # print(count_train)
        for batch_idx, (premise, hyphotheses, labels,  lenghts_promise, lenghts_hypotheses) in enumerate(self.train_loader):
            # print(batch_idx)
            premises = premise.to(self.device)
            hyphotheses = hyphotheses.to(self.device)
            labels = labels.to(self.device)
            lenghts_promise = lenghts_promise.to(self.device)
            lenghts_hypotheses = lenghts_hypotheses.to(self.device)

            output = self.model(premises, hyphotheses, lenghts_promise, lenghts_hypotheses, self.device)  # [batch_size, num_classes]
            self.optimizer.zero_grad()
            loss = self.loss_func(output, labels)
            loss.backward()
            self.optimizer.step()

            pred = torch.max(output, 1)[1]
            correct_num += torch.sum(torch.eq(pred, labels)).item()
            # print(correct_num)
            train_loss += loss.item()
        train_loss = train_loss / count_train
        train_acc = correct_num / count_train
        return train_loss, train_acc

    def evaluate(self, mode):
        self.model.eval()
        total_loss = 0.0
        correct_num = 0
        if mode == 'dev':
            data_loader = self.dev_loader
            count = len(self.snli_dataset.dev_data['labels'])
        elif mode == 'test':
            data_loader = self.test_loader
            count = len(self.snli_dataset.test_data['labels'])
        else:
            self.logger.info('mode is error')
            return
        # count = len(self.snli_dataset.dev_data)
        with torch.no_grad():
            for batch_idx, (premise, hyphotheses, labels,  lenghts_promise, lenghts_hypotheses) in enumerate(data_loader):
                premises = premise.to(self.device)
                hyphotheses = hyphotheses.to(self.device)
                labels = labels.to(self.device)
                lenghts_promise = lenghts_promise.to(self.device)
                lenghts_hypotheses = lenghts_hypotheses.to(self.device)

                output = self.model(premises, hyphotheses, lenghts_promise, lenghts_hypotheses, self.device)   # [batch_size, num_classes]
                loss = self.loss_func(output, labels)

                pred = torch.argmax(output, 1)
                correct_num += torch.sum(torch.eq(pred, labels)).item()
                total_loss += loss.item()

        return total_loss / count, correct_num / count

    def test(self):
        self.model = self.load_model()
        test_loss, test_acc = self.evaluate('test')
        return test_loss, test_acc

    def save_model(self, epoch_cur):
        self.logger.info('save the best model from epoch: {}'.format(epoch_cur))
        torch.save(self.model.state_dict(), self.config.model_dir)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.config.model_dir))
        return self.model
