# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-31 11:56
@Auth ： songxinxin
@File ：main.py
"""
from config import praser
from train import Trainer


if __name__ == '__main__':
    config = praser()
    Trainer = Trainer(config)
    if not config.test:
        Trainer.train()
    else:
        Trainer.test()