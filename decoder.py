import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        concatenated_hidden_dim = hidden_dim * 3
        self.decoder = nn.Linear(in_features=concatenated_hidden_dim, out_features=1)

    def forward(self, node_hidden, edge_hidden, data):
        edge_src_idx = data.edge_index[0]
        edge_dst_idx = data.edge_index[1]

        edge_src_hidden = node_hidden[edge_src_idx]
        edge_dst_hidden = node_hidden[edge_dst_idx]
        concatenated_hidden = torch.cat(
            [edge_hidden, edge_src_hidden, edge_dst_hidden], dim=-1
        )

        return self.decoder(concatenated_hidden)
