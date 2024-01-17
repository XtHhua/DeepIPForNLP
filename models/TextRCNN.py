# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""Recurrent Convolutional Neural Networks for Text Classification"""


class Model(nn.Module):
    def __init__(
        self,
        vocab_size=21128,
        embed=768,
        hidden_size=256,
        num_layers=1,
        dropout=0.5,
        pad_size=32,
        num_classes=10,
    ):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.embed = embed
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.pad_size = pad_size
        self.num_classes = num_classes
        self.embedding = nn.Embedding(self.vocab_size, self.embed)
        self.lstm = nn.LSTM(
            self.embed,
            self.hidden_size,
            self.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=self.dropout,
        )
        self.maxpool = nn.MaxPool1d(self.pad_size)
        self.fc = nn.Linear(self.hidden_size * 2 + self.embed, self.num_classes)

    def forward(self, x):
        embed = self.embedding(
            x["input_ids"]
        )  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out
