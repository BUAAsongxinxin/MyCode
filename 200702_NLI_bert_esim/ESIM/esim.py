# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-10 15:54
@Auth ： songxinxin
@File ：esim.py
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from ESIM.utils import *


class Esim(nn.Module):
    def __init__(self, congifs, vocab_size, embedding):
        super(Esim, self).__init__()
        self.config = congifs
        self.embedding = embedding  # 需要load一下矩阵
        self.vocab_size = vocab_size
        # self.max_premise_len = max_premise_len
        # self.max_hypothesis_len = max_hypothesis_len
        self.embedding_dim = congifs.embedding_dim
        self.hidden_dim = congifs.hidden_dim
        self.num_layers = congifs.num_layers
        self.num_classes = congifs.num_classes
        self.dropout = congifs.dropout_fc

        self.inputEncoding = InputEncoding(self.embedding, self.vocab_size, self.embedding_dim, self.hidden_dim, self.num_layers)

        self.local_infer_Model = LocalInferenceModeling()
        self.infer_com = InferenceComposition(self.hidden_dim, self.num_layers)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 8, self.hidden_dim *4),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, premises, hypotheses, lengths_promise, lengths_hypothesis, device):
        # premise:[batch_size, seq_len1]
        # hypotheses: [batch_size, seq_len2]
        premises_mask = get_mask(premises, lengths_promise)
        hypotheses_mask = get_mask(hypotheses, lengths_hypothesis)
        # Input Encoding
        premises_en = self.inputEncoding(premises, lengths_promise, device)
        hypotheses_en = self.inputEncoding(hypotheses, lengths_hypothesis, device)

        # Local Inference Modeling
        premises_local, hypotheses_local = self.local_infer_Model(premises_en, hypotheses_en, premises_mask, hypotheses_mask, device)
        # premise_local:[batch_size, len1, hidden_dim * 8], hypo_local:[batch_size, len2, hidden_dim * 8]

        # Inference Compotition
        premises_com = self.infer_com(premises_local, lengths_promise, device)
        hypotheses_com = self.infer_com(hypotheses_local, lengths_hypothesis, device)  # [batch_size, seq_len, hidden_dim * 2]

        # pool
        premises_pool = pool(premises_com)   # [batch_size, hidden_dim * 4]
        hypotheses_pool = pool(hypotheses_com)  # [batch_size, hidden_dim * 4]
        fc_input = torch.cat([premises_pool, hypotheses_pool], dim=-1)  # [batch_size, hidden_dim * 8]

        # fc
        logits = self.fc(fc_input)
        return logits


class InputEncoding(nn.Module):  # Embedding, BiLSTM
    def __init__(self, embedding, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(InputEncoding, self).__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size
        print(embedding.shape)
        if self.embedding is not None:
            self.embed = nn.Embedding.from_pretrained(embedding)
            self.embedding_dim = embedding.shape[1]
        else:
            self.embed = nn.Embedding(vocab_size, embedding_dim)
            self.embedding_dim = embedding_dim

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)

    def forward(self, x, lengths, device):
        #  data[batch_size, seq_len]
        embed_x = self.embed(x)  # [batch_size, seq_len, embed_dim]

        sorted_seq, sorted_len, sorted_index, reorder_index = sorted_by_len(embed_x, lengths, device)
        # print(sorted_seq.shape)
        # print(sorted_len)
        packed_x = nn.utils.rnn.pack_padded_sequence(sorted_seq, sorted_len, batch_first=True)

        # print(packed_x.shape)
        out, _ = self.lstm(packed_x)  # [batch_size, seq_len, hidden_dim * 2]
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        reorder_output = torch.index_select(out, dim=0, index=reorder_index)

        return reorder_output


class LocalInferenceModeling(nn.Module):
    def __init__(self):
        super(LocalInferenceModeling, self).__init__()

    def forward(self, premise, hypothesis, premise_mask, hypothesis_mask, device):
        # premise: [batch_size, seq_len_1, hidden_dim*2]
        # hypothesis:[batch_size, seq_len_2, hidden_dim*2]
        similarity_matrix = torch.bmm(premise, hypothesis.transpose(2, 1).contiguous())  # [batch_size, len1, len2]

        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask, device)  # [batch_size, len1, len2]
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(),premise_mask, device)  # [batch_size, len2, len1]

        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        attended_premises = weighted_sum(hypothesis, prem_hyp_attn, premise_mask, device)  # [B, len1, hidden*2]
        attended_hypotheses = weighted_sum(premise, hyp_prem_attn, hypothesis_mask, device)  # [B, len2, hidden*2]

        compose_premise = submul(premise, attended_premises)  # [batch_size, len1, hidden_dim * 8]
        compose_hypothesis = submul(hypothesis, attended_hypotheses)  # [batch_size, len2, hidden_dim * 8]

        return compose_premise, compose_hypothesis

        # 最简单，不带mask的版本
        # weight_1 = F.softmax(similarity_matrix, dim=1)  # [batch_size, len1, len2]
        # align_premise = torch.bmm(weight_1, hypothesis)  # [batch_size, len1, hidden_dim * 2]
        #
        # weight_2 = F.softmax(similarity_matrix.transpose(1, 2), dim=-1)  # [batch_size, len1, len2]
        # align_hypothesis = torch.bmm(weight_2, premise)  # [batch_size, len2, hidden_dim * 2]
        #
        # compose_premise = submul(premise, align_premise)  # [batch_size, len1, hidden_dim * 8]
        # compose_hypothesis = submul(hypothesis, align_hypothesis)  # [batch_size, len2, hidden_dim * 8]
        #
        # return compose_premise, compose_hypothesis


class InferenceComposition(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(InferenceComposition, self).__init__()
        self.input_size = hidden_dim * 8
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x, lengths, device):
        sorted_seq, sorted_len, sorted_index, reorder_index = sorted_by_len(x, lengths, device)
        packed_x = nn.utils.rnn.pack_padded_sequence(sorted_seq, sorted_len, batch_first=True, enforce_sorted=False)

        out, _ = self.lstm(packed_x)  # [batch_size, seq_len, hidden_dim * 2]
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        reorder_output = torch.index_select(out, dim=0, index=reorder_index)
        # out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim * 2]
        return reorder_output
