import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.node_encoder = nn.Linear(in_features=4, out_features=hidden_dim)
        self.edge_encoder = nn.Linear(in_features=1, out_features=hidden_dim)

    def forward(self, data, read_data_batched):
        node_hidden = self.node_encoder(read_data_batched)
        edge_hidden = self.edge_encoder(data.overlap_similarity)

        return (node_hidden, edge_hidden)
