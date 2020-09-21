# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-07 15:19
@Auth ： songxinxin
@File ：main.py
"""
from bert.train import *
from ESIM.train_esim import *
from config import *


if __name__ == '__main__':
    config = config()
    trainer = None
    if config.model == 'bert':
        trainer = Trainer(config)
    elif config.model == 'esim':
        trainer = TrainerEsim(config)
    else:
        print("mode is unexisted")

    if not config.test:
        trainer.train()
    else:
        trainer.test()
