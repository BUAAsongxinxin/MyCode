# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-21 21:54
@Auth ： songxinxin
@File ：model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def argmax(vec):
    #  return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# 计算log∑e的xi次方，前向算法需要用到，做减法的原因在于减去最大值可以避免e的指数次，计算机上溢？
# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score)))


class BiLSTM_CRF(nn.Module):
    def __init__(self, config, vocab_size, tag_to_idx, device):
        super(BiLSTM_CRF, self).__init__()
        self.config = config
        self.device = device
        self.vocab_size = vocab_size
        self.tag_to_idx = tag_to_idx
        self.tagset_size = len(tag_to_idx)
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim

        # word_embedding + lstm
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=2, bidirectional=True, dropout=0.3)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)  # lstm_out to tag
        # self.hidden2tag = nn.Sequential(nn.Linear(self.hidden_dim, self.tagset_size),
        #                                 nn.Softmax())

        # CRF
        # 状态转移矩阵，transitions[i,j]表示状态j到状态i的score
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size, device=self.device))
        self.transitions.data[tag_to_idx['START'], :] = -10000  # 不会有状态转移到start
        self.transitions.data[:, tag_to_idx['STOP']] = -10000  # STOP不会转移到任何状态

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2, device=self.device), torch.randn(2, 1, self.hidden_dim // 2, device=self.device))  # lstm的hidden状态


    # 找出概率最大的路径的分数，使用的是动态规划的思想
    def _forward_arg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000., device=self.device)  # [1, tagset_size]

        # 初始的时候Start_tag = 0  START到任何tag的值都为0，表示开始传播
        init_alphas[0][self.tag_to_idx['START']] = 0.

        # 赋值给变量方便后向传播, forward_var是之前步骤的score
        forward_var = init_alphas

        # 开始迭代
        for feat in feats:  # feat:[tagset_size] 每个word可能的label，对seq的每个word进行遍历
            alpha_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):  # 这一轮迭代：所有其他标签到这个词的概率
                # 状态特征函数得分,feat是emission matrix
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)  # [1, tagset_size]  表示next_tag为label[i]的emission score
                # 状态转移函数得分,其他状态转移到状态next_tag的得分
                trans_score = self.transitions[next_tag].view(1, -1)  # [1, tagset_size]  trans_score[0,i]表示第i个tag转移到next_tag的score

                next_tag_var = forward_var + trans_score + emit_score  # [1, tagset_size] next_tag_var[0,i]表示第i个tag到next_tag的整条路径的分数

                alpha_t.append(log_sum_exp(next_tag_var).view(1))  # 到next_tag的最好路径的score，for执行完之后是一个长为tagsize的数组
            forward_var = torch.cat(alpha_t).view(1, -1)  # [1, tagset_size] forward——var[0][i]当前word到tag[i]的最好的得分

        terminal_var = forward_var + self.transitions[self.tag_to_idx['STOP']]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):  # gives a score of a provided tag squence 根据真实标签计算的score
        score = torch.zeros(1, device=self.device)
        tags = torch.cat([torch.tensor([self.tag_to_idx['START']], dtype=torch.long, device=self.device), tags])
        for i, feat in enumerate(feats):
            score += self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]

        score += self.transitions[self.tag_to_idx['STOP'], tags[-1]]
        return score

    def _get_lstm_features(self, sentence):  # 得到feats[seq_len, tagsize]就是发射矩阵
        self.hidden = self.init_hidden()
        embeds = self.word_embedding(sentence).view(len(sentence), 1, -1)  # [seq_len, embedding_dim] -> [seq_len, 1, embedding_dim]
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)  # out:[seq_len, 1, hidden_dim], hidden:(h,c)(num_layers*2, 1, hidden_dim//2)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)  # [seq_len, hidden_dim]
        lstm_feats = self.hidden2tag(lstm_out)  # [seq_len, tagsize]
        # lstm_feats = F.softmax(lstm_feats, dim=1)
        # print(lstm_feats.shape)
        print(lstm_feats[0])

        return lstm_feats

    # 解码得到预测的序列，以及预测的得分，预测的时候使用
    def _viterbi_decode(self, feats):  # feats:[seq_len, tagset_size]
        backpointers = []

        # 初始化
        init_vvars = torch.full((1, self.tagset_size), -10000, device=self.device)
        init_vvars[0][self.tag_to_idx['START']] = 0

        # 步骤i的forward_var保留步骤i-1的viterbi变量
        forward_var = init_vvars
        for feat in feats:  # feat:[1, tagset size] word的每一个可能的label
            bptrs_t = []  # holds the breakpointers for this step,即当前到所有tag的最大值索引
            viterbivars_t = []  # holds the viterbi variables for this step，当前word到所有tag的最大分数

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the previous step,
                # plus the score of transitioning from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)  # 选最大
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_idx['STOP']]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]  # 开始记录分数

        # Follow the back pointers to decode the best path  根据\delta找最大路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # Pop off the start tag
        start = best_path.pop()
        assert start == self.tag_to_idx['START']
        best_path.reverse()

        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)  # [seq_len, tag_size]
        forward_score = self._forward_arg(feats)
        gold_score = self._score_sentence(feats, tags)  # 根据两者之间的差值进行反向传播
        return forward_score - gold_score

    def forward(self, sentence):  # 只是用来预测
        lstm_feats = self._get_lstm_features(sentence)  # 发射矩阵
        score, tag_seq = self._viterbi_decode(lstm_feats)  # find the best path
        return score, tag_seq

