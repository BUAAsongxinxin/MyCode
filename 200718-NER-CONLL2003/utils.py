# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-21 16:35
@Auth ： songxinxin
@File ：utils.py
"""
import logging
import torch

# 定义logger
def log(log_file_path):
    log = logging.getLogger()
    log.setLevel(logging.INFO)

    # 写入日志文件
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)

    # 输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 定义输出格式
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 添加到log
    log.addHandler(fh)
    log.addHandler(ch)

    return log


tags = [(0,1),(2,3),(4,5),(6,7)]

def _find_tag(labels, B_label = 1, I_label = 2):
    result = []
    for num in range(len(labels)):
        if labels[num] == B_label:
            song_pos = num
            length = 1
            for num2 in range(num, len(labels)):
                if labels[num2] == I_label:
                    length += 1
                else:
                    result.append((song_pos, length))
                    break
            num = num2
    return result

def find_all_tag(labels):
    result = {}
    for tag in tags:
        res = _find_tag(labels, B_label = tag[0], I_label = tag[1])
        result[tag[0]] = res
    return result

def precision(pre_labels, true_labels):
    pre = []
    pre_result = find_all_tag(pre_labels)
    for name in pre_result:
        for x in pre_result[name]:
            if x:
                if pre_labels[x[0]:x[0]+x[1]] == true_labels[x[0]:x[0]+x[1]]:
                    pre.append(1)
                else:
                    pre.append(0)
    return sum(pre)/len(pre) if len(pre) != 0 else 0

def recall(pre_labels, true_labels):
    re = []
    true_result = find_all_tag(true_labels)
    for name in true_result:
        for x in true_result[name]:
            if x:
                if pre_labels[x[0]:x[0]+x[1]] == true_labels[x[0]:x[0]+x[1]]:
                    re.append(1)
                else:
                    re.append(0)
    return sum(re)/len(re) if len(re) != 0 else 0


def calculate_score(true, pre, num_classes):
    prec = precision(pre, true)
    reca = recall(pre, true)
    return prec, reca, (2 * prec * reca) / (prec + reca) if (prec + reca) != 0 else 0

