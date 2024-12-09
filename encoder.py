import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.node_encoder = nn.Linear(in_features=4, out_features=hidden_dim)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.edge_encoder = nn.Linear(in_features=1, out_features=hidden_dim)

    def forward(self, data, read_data_batched):
        node_hidden = self.node_encoder(read_data_batched)
        # Collapse read information into single embedding along sequence dimension (-2)
        (_, node_hidden) = self.gru(node_hidden)
        # Remove GRU's D * num_layers dimension
        # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU
        node_hidden = torch.squeeze(node_hidden)

        # Need to add extra dimension to overlap_similarity to treat it like a 1-dim feature
        edge_hidden = self.edge_encoder(data.edge_attr)

        return (node_hidden, edge_hidden)
