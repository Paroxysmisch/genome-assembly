import torch
from torch import nn

import layers

class SymGatedGCNModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, num_layers, hidden_edge_scores, batch_norm, dropout=None):
        super().__init__()
        hidden_node_features = hidden_edge_features
        self.linear1_node = nn.Linear(node_features, hidden_node_features)
        self.linear2_node = nn.Linear(hidden_node_features, hidden_features)
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features) 
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features) 
        self.gnn = layers.SymGatedGCN_processor(num_layers, hidden_features, batch_norm, dropout=dropout)
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e):
        x = self.linear1_node(x)
        x = torch.relu(x)
        x = self.linear2_node(x)

        e = self.linear1_edge(e)
        e = torch.relu(e)
        e = self.linear2_edge(e)
        
        x, e = self.gnn(graph, x, e)
        scores = self.predictor(graph, x, e)
        return scores
