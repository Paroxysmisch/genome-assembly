import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, hidden_dim, gru_dim=8):
        super().__init__()
        self.node_encoder = nn.Linear(in_features=4, out_features=gru_dim)
        self.gru = nn.GRU(input_size=gru_dim, hidden_size=gru_dim, batch_first=True)
        self.gru_expander = nn.Linear(in_features=gru_dim, out_features=hidden_dim)
        self.edge_encoder = nn.Linear(in_features=1, out_features=hidden_dim)

    def forward(self, data, read_data_batched):
        node_encoded = self.node_encoder(read_data_batched).squeeze(0)
        num_nodes, seq_len, hidden_dim = node_encoded.size()
        # Collapse read information into single embedding along sequence dimension (-2)
        # Chunk the GRU inputs to reduce memory usage
        chunk_size = 4096
        node_hidden = torch.zeros((1, num_nodes, hidden_dim), device=node_encoded.get_device())
        for i in range(0, seq_len, chunk_size):
            chunk = node_encoded[:, i:i + chunk_size, :]
            (_, node_hidden) = self.gru(chunk, node_hidden)
        # Remove GRU's D * num_layers dimension
        # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU
        node_hidden = torch.squeeze(node_hidden)
        node_hidden = self.gru_expander(node_hidden)

        # Need to add extra dimension to overlap_similarity to treat it like a 1-dim feature
        edge_hidden = self.edge_encoder(data.edge_attr)

        return (node_hidden, edge_hidden)
