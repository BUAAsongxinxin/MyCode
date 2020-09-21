# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-15 11:29
@Auth ： songxinxin
@File ：utils.py
"""
import torch
import torch.nn.functional as F
import logging


def pool(x):
    x = x.transpose(1, 2)  # [batch_size, hidden_dim*2, seq_len]
    avg_pool = F.avg_pool1d(x, x.shape[-1]).squeeze(-1)  # [batch_size, hidden_dim*2]
    max_pool = F.max_pool1d(x, x.shape[-1]).squeeze(-1)

    return torch.cat([avg_pool, max_pool], dim=-1)  # [batch_size, hidden_dim * 4]


def submul(a, align_a):
    sub = a - align_a
    mul = a * align_a

    return torch.cat([a, align_a, sub, mul], dim=-1)  # [batch_size, len, hidden_dim * 8]


def sorted_by_len(batch_seq, seq_len, device):
    """
    重新排序，用于nn.utils.rnn.pad_packed_sequence
    :param batch_seq:
    :param seq_len:
    :return:
    seq_sorted: A tensor containing the input batch reordered by
            sequences lengths.
    sorted_len: A tensor containing the sorted lengths of the
        sequences in the input batch.
    sorted_index: A tensor containing the indices used to permute the input
        batch in order to get 'sorted_batch'.
    reordered_index: A tensor containing the indices that can be used to
        restore the order of the sequences in 'sorted_batch' so that it
        matches the input batch.

    """
    sorted_len, sorted_index = torch.sort(seq_len, dim=0, descending=True)
    seq_sorted = torch.index_select(batch_seq, 0, sorted_index)

    idx_range = torch.arange(0, len(sorted_index), device=device)
    _, reverse_map = torch.sort(sorted_index, 0, descending=False)

    reorder_index = torch.index_select(idx_range, 0, reverse_map)

    return seq_sorted, sorted_len, sorted_index, reorder_index


def get_mask(batch_seq, seq_lens):
    # get mask:pad的位置为0
    batch_size = batch_seq.shape[0]
    max_length = torch.max(seq_lens)

    mask = torch.ones(batch_size, max_length, dtype=torch.float)
    mask[batch_seq[:, :max_length] == 0] = 0.0
    # 不同的batch，max_length不一样，而pad的时候是总的max_length，要加:max_length

    return mask


def masked_softmax(tensor, mask, device):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).

    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension. [batch_size, len1, len2]
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else. [batch_size]

    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1]).to(device)  # [batch_size*len1, len2]

    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()  # broadcast  [batch_size, len1, len2]
    reshaped_mask = mask.view(-1, mask.size()[-1]).to(device)  # [batch_size*len1, len2]

    result = F.softmax(reshaped_tensor * reshaped_mask, dim=-1)  # [batch_size*len1, len2] softmax之前对应元素设置为0
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)

    return result.view(*tensor_shape)


def weighted_sum(tensor, weights, mask, device):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.

    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.

    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)  # [B, len1/2, hidden*2]

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)  # [B, 1, len1/2]
    mask = mask.transpose(-1, -2)  # [B, len1/2, 1]
    mask = mask.expand_as(weighted_sum).contiguous().float().to(device)  # [B, len1/2, hidden*2]

    return weighted_sum * mask


def log(log_file_path):
    # 创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # log等级总开关

    # 创建一个handler, 用于写入日志文件
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关

    # 创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 将logger添加到handler里面
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

