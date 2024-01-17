# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

"""Convolutional Neural Networks for Sentence Classification"""


class Model(nn.Module):
    def __init__(
        self,
        vocab_size=21128,
        embed=768,
        num_filters=256,
        filter_sizes=(2, 3, 4),
        drop_out=0.5,
        num_classes=10,
    ):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.embed = embed
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        self.embedding = nn.Embedding(self.vocab_size, self.embed)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.embed)) for k in self.filter_sizes]
        )
        self.dropout = nn.Dropout(drop_out)
        self.lin = nn.Linear(
            self.num_filters * len(self.filter_sizes), self.num_classes
        )

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x["input_ids"])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.lin(out)
        return out
