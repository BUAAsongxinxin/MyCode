# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-21 22:39
@Auth ： songxinxin
@File ：main.py
"""
from bert.train_bert import *
from bilstm_crf.train import BiLSTMCRFTrainer
from config import config

if __name__ == '__main__':
    config = config()

    if config.model == 'bert':
        trainer = BertTrainer(config)
    elif config.model == 'bilstm_crf':
        trainer = BiLSTMCRFTrainer(config)
    else:
        pass

    if not config.test:
        trainer.train()
    else:
        trainer.test()
