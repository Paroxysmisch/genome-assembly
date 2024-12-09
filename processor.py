import torch.nn as nn
from torch_geometric.nn import Sequential, GCNConv

class GCN_Processor(nn.Module):
    def __init__(self, num_layers, hidden_dim):
        super().__init__()

        convs = []
        for _ in range(num_layers - 1):
            convs.append((GCNConv(hidden_dim, hidden_dim), 'x, edge_index -> x'))
            convs.append(nn.ReLU(inplace=True))
        convs.append((GCNConv(hidden_dim, hidden_dim), 'x, edge_index -> x'))
        self.convs = Sequential('x, edge_index', convs)

    def forward(self, node_hidden, edge_hidden, data):
        node_hidden = self.convs(x=node_hidden, edge_index=data.edge_index)
        return node_hidden, edge_hidden
