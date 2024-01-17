# coding: UTF-8
import torch.nn as nn
from transformers import BertForSequenceClassification


class Model(nn.Module):
    def __init__(
        self,
        classes=10,
        hidden_size=768,
    ):
        super(Model, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            "bert-base-chinese", num_labels=hidden_size
        )
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lin = nn.Linear(hidden_size, classes)

    def forward(self, x):
        sequence_classifier_output = self.bert(**x)
        out = self.lin(sequence_classifier_output.logits)
        return out
