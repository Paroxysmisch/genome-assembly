import torch.nn as nn
from torch_geometric.nn import conv


class GCN_Processor(nn.Module):
    def __init__(self, num_layers, hidden_dim):
        super().__init__()

        convs = []
        for _ in range(num_layers - 1):
            convs.append(conv.GCNConv(hidden_dim, hidden_dim))
            convs.append(nn.ReLU())
        convs.append(conv.GCNConv(hidden_dim, hidden_dim))
        self.convs = nn.Sequential(*convs)

    def forward(self, node_hidden, edge_hidden, data):
        node_hidden = self.convs(x=node_hidden, edge_index=data.edge_index)
        return node_hidden, edge_hidden
