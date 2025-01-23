import torch
from torch import nn

class NodeEdgeEncoder(nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_intermediate_hidden_features, num_hidden_features):
        super().__init__()
        self.linear1_node = nn.Linear(num_node_features, num_intermediate_hidden_features)
        self.linear2_node = nn.Linear(num_intermediate_hidden_features, num_hidden_features)
        self.linear1_edge = nn.Linear(num_edge_features, num_intermediate_hidden_features) 
        self.linear2_edge = nn.Linear(num_intermediate_hidden_features, num_hidden_features) 

    def forward(self, x, e):
        x = self.linear1_node(x)
        x = torch.relu(x)
        x = self.linear2_node(x)

        e = self.linear1_edge(e)
        e = torch.relu(e)
        e = self.linear2_edge(e)

        return x, e


class NodeEdgeReadsEncoder(nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_intermediate_hidden_features, num_hidden_features, num_gru_features):
        super().__init__()
        num_linear2_node_output_features = num_hidden_features // 2
        self.linear1_node = nn.Linear(num_node_features, num_intermediate_hidden_features)
        self.linear2_node = nn.Linear(num_intermediate_hidden_features, num_linear2_node_output_features)
        self.linear1_edge = nn.Linear(num_edge_features, num_intermediate_hidden_features) 
        self.linear2_edge = nn.Linear(num_intermediate_hidden_features, num_hidden_features) 
        self.gru = nn.GRU(input_size=4, hidden_size=num_gru_features, batch_first=True)
        self.gru_expander = nn.Linear(in_features=num_gru_features, out_features=num_hidden_features - num_linear2_node_output_features)
        self.num_gru_features = num_gru_features

    def forward(self, graph, x, e):
        x = self.linear1_node(x)
        x = torch.relu(x)
        x = self.linear2_node(x)

        e = self.linear1_edge(e)
        e = torch.relu(e)
        e = self.linear2_edge(e)

        # Collapse read information into single embedding along sequence dimension (-2)
        # Chunk the GRU inputs to reduce memory usage
        read_data = graph.ndata['read_data']
        num_nodes, seq_len = read_data.shape[:2]
        chunk_size = 4096
        read_hidden = torch.zeros(
            (1, num_nodes, self.num_gru_features), device=x.device
        )
        read_data_padds = read_data.sum(-1)
        read_data_lens = read_data_padds.sum(-1).long()-1
        arranged = torch.arange(read_data_lens.shape[0], device=read_data_lens.device)
        (rh, rho) = self.gru(read_data, read_hidden)
        # for i in range(0, seq_len, chunk_size):
        #     chunk = read_data[:, i : i + chunk_size, :]
        #     (_, read_hidden_residual) = self.gru(chunk, read_hidden)
        #     read_hidden = read_hidden_residual + read_hidden

        # Remove GRU's D * num_layers dimension
        # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU
        # read_hidden = torch.squeeze(read_hidden)
        read_hidden = rh[arranged, read_data_lens, :]
        read_hidden = self.gru_expander(read_hidden)

        x = torch.concatenate([x, read_hidden], dim=-1)

        return x, e
