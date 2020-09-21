# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-07 15:19
@Auth ： songxinxin
@File ：bert.py
"""
import torch.nn as nn
from transformers import BertModel


class BertNLi(nn.Module):
    def __init__(self, config):
        super(BertNLi, self).__init__()
        self.config = config

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout_emb = nn.Dropout(config.dropout_emb)
        self.fc = nn.Linear(in_features=768, out_features=config.num_classes)
        self.dropout_fc = nn.Dropout(config.dropout_fc)

        unfreeze_layers = ['layer.10', 'layer.11', 'bert.pooler']
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

        # for param in self.bert.base_model.parameters():  # 固定参数
        #     param.requires_grad = False

    def forward(self, input_ids, token_type_ids, attention_mask):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)  # [batch_size, hidden_dim]
        pooled_output = self.dropout_emb(pooled_output)

        fc_out = self.fc(pooled_output)  # [batch_size, num_classes]
        return fc_out
