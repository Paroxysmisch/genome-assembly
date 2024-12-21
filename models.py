from enum import Enum

import torch
from torch import nn

import layers


class SymGatedGCNModel(nn.Module):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        num_intermediate_hidden_features,
        num_hidden_features,
        num_layers,
        num_hidden_edge_scores,
        batch_norm,
        dropout=None,
    ):
        super().__init__()
        self.linear1_node = nn.Linear(
            num_node_features, num_intermediate_hidden_features
        )
        self.linear2_node = nn.Linear(
            num_intermediate_hidden_features, num_hidden_features
        )
        self.linear1_edge = nn.Linear(
            num_edge_features, num_intermediate_hidden_features
        )
        self.linear2_edge = nn.Linear(
            num_intermediate_hidden_features, num_hidden_features
        )
        self.gnn = layers.SymGatedGCN_processor(
            num_layers, num_hidden_features, batch_norm, dropout=dropout
        )
        self.predictor = layers.ScorePredictor(
            num_hidden_features, num_hidden_edge_scores
        )

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


class SymGatedGCNWithReadsModel(nn.Module):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        num_intermediate_hidden_features,
        num_hidden_features,
        num_layers,
        num_hidden_edge_scores,
        batch_norm,
        dropout=None,
    ):
        super().__init__()
        self.encoder = layers.NodeEdgeReadsEncoder(
            num_node_features,
            num_edge_features,
            num_intermediate_hidden_features,
            num_hidden_features,
            num_gru_features=8,
        )
        self.gnn = layers.SymGatedGCN_processor(
            num_layers, num_hidden_features, batch_norm, dropout=dropout
        )
        self.predictor = layers.ScorePredictor(
            num_hidden_features, num_hidden_edge_scores
        )

    def forward(self, graph, x, e):
        x, e = self.encoder(graph, x, e)
        x, e = self.gnn(graph, x, e)
        scores = self.predictor(graph, x, e)

        return scores


class ModelType(Enum):
    SymGatedGCN = SymGatedGCNModel
    SymGatedGCNReads = SymGatedGCNWithReadsModel
