## 自然语言推理（Natural Language Inference, NLI）

### 1. 任务描述

给定前提premise和假设hypothesis，判断p和h的关系:

- neural : 不相干
- controdiction : 冲突，即p和h有矛盾
- entailment : 蕴含，能从p推出h或者两者表达的是一个意思

| Premise                                                      | Hypothesis               | Judgment      |
| ------------------------------------------------------------ | ------------------------ | ------------- |
| A woman with a green headscarf , blue shirt and a very big grin | The woman is very happy. | entailment    |
| A woman with a green headscarf , blue shirt and a very big grin | The woman is young.      | neutral       |
| A woman with a green headscarf , blue shirt and a very big grin | The woman has been shot  | contradiction |

为什么要研究NLI？

- 自然语言推理是一个分类任务，使用准确率就可以客观有效的评价模型的好坏。

- 这样我们就可以**专注于语义理解和语义表示**，如果这部分做得好的话，例如可以生成很好的句子表示的向量，那么我们就可以将这部分成果轻易迁移到其他任务中，例如对话，问答等。

- **简单的评价标准、专注于语义理解、研究成果具有高迁移性。**

  

### 2.数据集

- [Stanford Natural Language Inference（SNLI）](https://nlp.stanford.edu/projects/snli/)

  NLI领域第一个大规模人工标注的数据集，包含 55w 条训练样本，1w条验证样本，1w条测试样本，每条样本是一个句子对

  - `annotator_labels`: 5个标注人的评估结果
  - `gold_label`: 要预测的标签，the majority of annotators
  - `sentence1`: premise
  - `sentence2`: hypothesis
  - `sentence{1,2}_parse`: The parse produced by the Stanford Parser (3.5.2, case insensitive PCFG, trained on the standard training set augmented with the parsed Brown Corpus) in Penn Treebank format.
  - `sentence{1,2}_binary_parse`: The same parse as in sentence{1,2}_parse, but formatted for use in **tree-structured neural networks** with no unary nodes and no labels.

  ```json
  {
  	"annotator_labels": ["neutral", "contradiction", "contradiction", "neutral", "neutral"],
  	"captionID": "2677109430.jpg#1",
  	"gold_label": "neutral",
  	"pairID": "2677109430.jpg#1r1n",
  	"sentence1": "This church choir sings to the masses as they sing joyous songs from the book at a church.",
  	"sentence1_binary_parse": "( ( This ( church choir ) ) ( ( ( sings ( to ( the masses ) ) ) ( as ( they ( ( sing ( joyous songs ) ) ( from ( ( the book ) ( at ( a church ) ) ) ) ) ) ) ) . ) )",
  	"sentence1_parse": "(ROOT (S (NP (DT This) (NN church) (NN choir)) (VP (VBZ sings) (PP (TO to) (NP (DT the) (NNS masses))) (SBAR (IN as) (S (NP (PRP they)) (VP (VBP sing) (NP (JJ joyous) (NNS songs)) (PP (IN from) (NP (NP (DT the) (NN book)) (PP (IN at) (NP (DT a) (NN church))))))))) (. .)))",
  	"sentence2": "The church has cracks in the ceiling.",
  	"sentence2_binary_parse": "( ( The church ) ( ( has ( cracks ( in ( the ceiling ) ) ) ) . ) )",
  	"sentence2_parse": "(ROOT (S (NP (DT The) (NN church)) (VP (VBZ has) (NP (NP (NNS cracks)) (PP (IN in) (NP (DT the) (NN ceiling))))) (. .)))"
  }
  ```

  

###3.ESIM

> ESIM是ACL2017的一篇论文，[Enhanced LSTM for Natural Language Inference](https://arxiv.org/abs/1609.06038v3)
>
> 重点：
>
> - 基于链式LSTMs精心设计了序列推断模型 (carefully designing sequential inference models based on chain LSTMs)；
> - 考虑局部推断和推断组合（in both local inference modeling and inference composition）

这样我们就可以专注于语义理解和语义表示，如果这部分做得好的话，例如可以生成很好的句子表示的向量，那么我们就可以将这部分成果轻易迁移到其他任务中，例如对话，问答等。

#### 3.1 模型架构

- 原论文图<img src="https://raw.githubusercontent.com/BUAAsongxinxin/picgo/master/img/20200712145304.png" alt="image-20200712145304156" style="zoom:50%;" />

- 清晰的版本<img src="https://raw.githubusercontent.com/BUAAsongxinxin/picgo/master/img/20200712145337.png" alt="image-20200712145337505" style="zoom: 50%;" />



#### 3.2 输入编码 Input Encoding

p和h分别接embedding层和双向LSTM层（为什么不用GRU，因为效果不好）

这里的 BiLSTM 是学习如何表示一句话中的 word 和它上下文的关系，可以理解成这是 在 word embedding 之后，在当前的语境下重新编码，得到新的 embeding 向量。

```python
def forward(self, x, lengths):
    # x:[batch_size, seq_len]
    embed_x = self.embed(x)  # [batch_size, seq_len, embed_dim]

    sorted_seq, sorted_len, sorted_index, reorder_index = sorted_by_len(embed_x, lengths)
    packed_x = nn.utils.rnn.pack_padded_sequence(sorted_seq, sorted_len, batch_first=True)

    out, _ = self.lstm(packed_x)  # [batch_size, seq_len, hidden_dim * 2]
    out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    reorder_output = torch.index_select(out, reorder_index)

    return reorder_output
```
####3.3 局部推理 Locality of inference

1. 首先对p和h进行 alignment（其实就是attention）

   计算相似度矩阵：<img src="https://raw.githubusercontent.com/BUAAsongxinxin/picgo/master/img/20200715215602.png" alt="image-20200712151959231" style="zoom:33%;" />

   使用相似度矩阵，互相生产彼此相似性加权后的句子，维度不变

   <img src="https://raw.githubusercontent.com/BUAAsongxinxin/picgo/master/img/20200712155701.png" alt="image-20200712152055138" style="zoom: 33%;" />

2. Enhancement of local Inference Information

   <img src="https://raw.githubusercontent.com/BUAAsongxinxin/picgo/master/img/20200712155651.png" alt="image-20200712155651762" style="zoom: 50%;" />

#### 3.4 inference composition

再一次用 BiLSTM 提前上下文信息，同时使用 MaxPooling 和 AvgPooling 进行池化操作, 最后接一个全连接层

```python
premises_pool = pool(premises_com)   # [batch_size, hidden_dim * 4]
hypotheses_pool = pool(hypotheses_com)  # [batch_size, hidden_dim * 4]
fc_input = torch.cat([premises_pool, hypotheses_pool], dim=-1)  # [batch_size, hidden_dim * 8]
# fc
logits = self.fc(fc_input)  # [batch_size, 3]
```
```python
def pool(x):
  x = x.transpose(1, 2)  # [batch_size, hidden_dim*2, seq_len]
  avg_pool = F.avg_pool1d(x, x.shape[-1]).squeeze(-1)  # [batch_size, hidden_dim*2]
  max_pool = F.max_pool1d(x, x.shape[-1]).squeeze(-1)
```

#### 3.5 结果

最简化的模型：3w条数据，acc在0.62，一个epoch8min左右

加了pad_packed,attetnion的mask，3w条数据，acc在0.67，一个epoch23min左右

#### 3.6 mask操作

- padding_mask: 在NLP任务中，由于句子长度不一致而进行padding操作，在sequence中加入零向量，RNN中mask步骤如下

  - 首先从dataloader中取出的数据是pad之后的

  - 使用torch.nn.utils.rnn.pack_padded_sequence,可以理解为压紧，注意是按列压
    - 按列压如图所示
    - 当enforce_sorted=True时，input必须是按lengths从大到小排列好的，the len of longest seq is lengths[0]，若设置了max_len, 则要注意lenghts中的len不能大于max_len，否则会报size的错误，最好还是按实际数据的最大长度来。
    - 返回PackedSquence,可直接输入rnn 

  <img src="https://raw.githubusercontent.com/BUAAsongxinxin/picgo/master/img/20200715210659.png" alt="y" style="zoom:33%;" />

  - 扔进rnn
  - 使用nn.utils.rnn.pad_packed_sequence()将得到的output填充回来
    - 是nn.utils.rnn.pack_padded_sequence的逆过程
    - 返回一个元组，包含被填充后的序列，和batch中序列的长度列表
  - 最后记得！！如果设置enforce_sorted为True时，记得将最后的output返回原来的顺序，以对应input的batch

- 关于attention的mask

  - key mask：在计算score之后，且softmax之前进行，将值设为0或很小的数字(如-e^12)，这样经过的softmax之后值几乎为0

  - query mask：在softmax之后进行，直接把对应元素设置为0即可

  ```python
  def masked_softmax(tensor, mask, device):
      tensor_shape = tensor.size()
      reshaped_tensor = tensor.view(-1, tensor_shape[-1]).to(device)  # [batch_size*len1, len2]
  
      # Reshape the mask so it matches the size of the input tensor.
      while mask.dim() < tensor.dim():
          mask = mask.unsqueeze(1)
      mask = mask.expand_as(tensor).contiguous().float()  # broadcast  [batch_size, len1, len2]
      reshaped_mask = mask.view(-1, mask.size()[-1]).to(device)  # [batch_size*len1, len2]
  
      result = F.softmax(reshaped_tensor * reshaped_mask, dim=-1)  # [batch_size*len1, len2] softmax之前对应元素设置为0
      result = result * reshaped_mask  # 对应元素设置为0
      # 1e-13 is added to avoid divisions by zero.
      result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
  
      return result.view(*tensor_shape)
  ```

- device：尤其对于一些中间定义的函数，注意在的device

### 4. Bert

#### 4.1 models

<img src="https://raw.githubusercontent.com/BUAAsongxinxin/picgo/master/img/20200712161245.png" alt="image-20200712161245033" style="zoom: 33%;" />

BERT接受句子对的训练，并期望使用1和0来区分这两个句子

- 参数:
  `input_ids`: (batch_size, seq_len) 代表输入实例的Tensor
  `token_type_ids=None`: (batch_size, sqe_len) 一个实例可以含有两个句子，这个相当于句子标记,用0和1来区分
  `attention_mask=None`: (batch_size*):  传入每个实例的长度，用于attention的mask，mask padded values
  `output_all_encoded_layers`=True: 控制是否输出所有encoder层的结果.
- 返回值:
  `encoded_layer`：长度为num_hidden_layers的(batch_size， sequence_length，hidden_size)的Tensor列表
  `pooled_output`: (batch_size, hidden_size), 最后一层encoder的第一个词[CLS]经过Linear层和激活函数Tanh()后的Tensor. 其代表了句子信息

```python
self.bert = BertModel.from_pretrained('bert-base-uncased')
```

`Uncased`表示在WordPiece标记化之前，文本已小写，例如，`John Smith`变为`john smith`。Uncased模型还会删除任何重音标记。`Cased`表示保留了真实的大小写和重音标记。通常，除非你知道案例信息对于你的任务很重要(例如，命名实体识别或词性标记)，否则`Uncased`模型会更好。

当使用`Uncased`的模型时，确保将--do_lower_case传递给示例训练脚本(如果使用自己的脚本，则将`do_lower_case=True`传递给FullTokenizer))

#### 4.2 一些结果

全部参数可调：1亿多参数，10w训练数据，25min/epoch

fixbert全部参数：768-3， 2307 个参数，1w数据，6min/epoch，acc0.47 左右

 微调最后两层参数：acc0.63左右，明显上升， 1w

 微调最后两层参数：acc0.67左右，明显上升， 3w

```python
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
```