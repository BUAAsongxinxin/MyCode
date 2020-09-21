# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-21 21:54
@Auth ： songxinxin
@File ：bert.py
"""
import torch.nn as nn
from transformers import BertModel


class BertNer(nn.Module):
    def __init__(self, config):
        super(BertNer, self).__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.hidden_dim = config.hidden_dim
        self.dropout_p = config.dropout_bert

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout_bert = nn.Dropout(self.dropout_p)
        self.fc = nn.Sequential(nn.Linear(768, self.hidden_dim * 4),
                                nn.Dropout(self.dropout_p),
                                nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
                                nn.Dropout(self.dropout_p),
                                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                                nn.Linear(self.hidden_dim, self.num_classes),
                                )

        # 冻结bert部分参数
        unfreeze_layers = ['layer.10', 'layer.11']
        for name, param in self.bert.named_parameters():  # name 类似：'encoder.layer.0.attention.self.query.weight'
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

    def forward(self, input_ids, token_type_ids, attention_mask):
        encoded, _ = self.bert(input_ids, token_type_ids, attention_mask)  # [batch_size, seq_len, hidden_size]
        encoded = self.dropout_bert(encoded)
        # 取出每一层的hiddenstate
        _, _, hidden_state = self.bert(input_ids, token_type_ids, attention_mask, output_hidden_states=True)  # [batch_size, seq_len, hidden_size]

        output = self.fc(encoded)  # [batch_size, seq_len, num_classes]
        return output

