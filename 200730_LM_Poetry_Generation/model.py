# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-30 17:53
@Auth ： songxinxin
@File ：model.py
"""
import torch.nn as nn
import torch


class PotryModel(nn.Module):
    def __init__(self, vocab_size, config, device, logger):
        super(PotryModel, self).__init__()
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.device = device
        self.batch_size = config.batch_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim=self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim * 2, vocab_size)

    def forward(self, text, hidden):
        embeds = self.embedding(text)  # [batch_size, seq_len, embedding_dim]
        output, hidden = self.gru(embeds, hidden)  # [batch_size, seq_len, hidden_dim*2], [num_layers*2, batch_size, hidden_dim]
        output = self.fc(output)  # [batch_size, seq_len, vocab_size]

        return output, hidden

    def init_hidden(self, seq_len):
        return torch.zeros((self.num_layers*2, seq_len, self.hidden_dim), device=self.device)




