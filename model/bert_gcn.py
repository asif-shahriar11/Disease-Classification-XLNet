import torch.nn as nn
import torch.nn.functional as F

from .bert import BERTClass
from .gcn import GCN

class BertGCN(nn.Module):
    def __init__(self, num_classes = 23, m = 0.7, gcn_layers = 2, gcn_hidden = 200, gcn_dropout = 0.5, gcn_norm = 'none'):
        super(BertGCN, self).__init__()
        self.m = m
        self.num_classes = num_classes
        self.bert = BERTClass()
        self.tokenizer = self.bert.get_tokenizer()
        self.gcn = GCN(input_features = 768, n_hidden = gcn_hidden, n_classes = num_classes, n_layers = gcn_layers - 1, activation = F.elu, dropout = gcn_dropout, normalization = gcn_norm)
        self.classifier = nn.Linear(768, num_classes)
        

    def forward(self, input_ids, attn_mask, token_type_ids, g, h):
        return NotImplementedError

    def get_tokenizer(self):
        return self.bert.get_tokenizer()

    def get_bert(self):
        return self.bert

    def get_gcn(self):
        return self.gcn