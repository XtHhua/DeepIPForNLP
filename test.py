'''
Descripttion: DeepModelIPProtection
version: 1.0
Author: XtHhua
Date: 2023-05-16 16:43:44
LastEditors: XtHhua
LastEditTime: 2023-11-05 12:29:12
'''
import torch
from transformers import (
    BertModel,
    BertTokenizer,
    BertForSequenceClassification,
    BertConfig,
)

# model = BertForSequenceClassification.from_pretrained("bert-base-chinese")
# print(model)


# 导出模型配置
# config = model.config
# print(type(config))
# print(config)
# config.save_pretrained("./source_model_info/")

# config = BertConfig.from_pretrained("./source_model_info/")
# print(type(config))
# print(config)

# # 加载预训练的BERT模型和词汇表
# model_name = "bert-base-chinese"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# vocab = tokenizer.get_vocab()
# print(len(vocab))
# model = BertModel.from_pretrained(model_name)

# # 输入文本
# text = "你好，世界！"

# # 对文本进行分词和编码
# tokens = tokenizer.tokenize(text)
# print(tokens)
# input_ids = tokenizer.convert_tokens_to_ids(tokens)
# print(input_ids)
# input_ids = torch.tensor([input_ids])

# # 获取文本的词嵌入向量
# outputs = model(input_ids)
# # print(outputs)
# embeddings = outputs.last_hidden_state
# print(type(embeddings))
# print(embeddings.shape)
# print(embeddings[0].shape)


# class Model(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.lin = torch.nn.Linear(4, 2)

#     def forward(self, x):
#         return self.lin(x)


# import torch.nn.utils.prune as prune

# x = torch.randn(4).unsqueeze(0)

# print(x)
# model = Model()
# print(model)
# print(f"weight:{model.lin.weight}")
# print(f"bias:{model.lin.bias}")

# y = model(x)
# print(f"before:{y}")
# pruner = prune.L1Unstructured(amount=0.5)
# pruner.apply(model.lin, name="weight", amount=0.5)
# print(model)
# print(f"weight:{model.lin.weight}")
# print(f"bias:{model.lin.bias}")
# prune.remove(model.lin, name="weight")
# print(f"weight:{model.lin.weight}")
# print(f"bias:{model.lin.bias}")
# y = model(x)
# print(f"after:{y}")


if __name__ == '__main__':
    from transformers import BertTokenizer, BertModel
    import torch

    # 加载 BERT tokenizer 和 BERT 模型
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertModel.from_pretrained("bert-base-chinese")
    text = "这是一些示例文本。"
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    print(input_ids)
    with torch.no_grad():
        output = model(input_ids)

    # 获取 token embeddings
    token_embeddings = output.last_hidden_state
    print(token_embeddings)
    print(token_embeddings.shape)


