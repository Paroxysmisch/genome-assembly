import torch.nn as nn
import layers

class SymGatedGCN_processor(nn.Module):
    def __init__(self, num_layers, hidden_features, batch_norm, dropout=None, rnf=False):
        super().__init__()
        self.convs = nn.ModuleList([
            layers.SymGatedGCN(hidden_features, hidden_features, batch_norm, dropout, rnf) for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for i in range(len(self.convs)):
            h, e = self.convs[i](graph, h, e)
        return h, e

class GAT_processor(nn.Module):
    def __init__(self, num_layers, hidden_features, batch_norm, dropout=None, rnf=False):
        super().__init__()
        self.convs = nn.ModuleList([
            layers.GAT(hidden_features, hidden_features, batch_norm, dropout, rnf) for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for i in range(len(self.convs)):
            h, e = self.convs[i](graph, h, e)
        return h, e

class SymGAT_processor(nn.Module):
    def __init__(self, num_layers, hidden_features, batch_norm, dropout=None, rnf=False):
        super().__init__()
        self.convs = nn.ModuleList([
            layers.SymGAT(hidden_features, hidden_features, batch_norm, dropout, rnf) for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for i in range(len(self.convs)):
            h, e = self.convs[i](graph, h, e)
        return h, e
