# coding: UTF-8
import torch.nn as nn

"""Recurrent Neural Network for Text Classification with Multi-Task Learning"""


class Model(nn.Module):
    def __init__(
        self,
        vocab_size=21128,
        embed=768,
        hidden_size=128,
        num_layers=2,
        dropout=0.5,
        num_classes=10,
    ):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.embed = embed
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.embedding = nn.Embedding(self.vocab_size, self.embed)
        self.lstm = nn.LSTM(
            self.embed,
            self.hidden_size,
            self.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(self.hidden_size * 2, self.num_classes)

    def forward(self, x):
        out = self.embedding(
            x["input_ids"]
        )  # [batch_size, seq_len, embeding]=[128, 32, 768]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out
