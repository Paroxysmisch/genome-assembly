from enum import Enum

import torch
from mamba_ssm import Mamba
from torch import nn

import layers

import sys
import os
submodule_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './mamba_pytorch/'))
sys.path.insert(0, submodule_path)
from mambapy.mamba import Mamba as MambaPytorch, MambaConfig


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
        rnf=False,
        granola=False,
        use_cuda=True,
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
            num_layers, num_hidden_features, batch_norm, dropout=dropout, rnf=rnf, granola=granola
        )
        self.predictor = layers.ScorePredictor(
            num_hidden_features * 3, num_hidden_edge_scores
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


class GATModel(nn.Module):
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
        rnf=False,
        granola=False,
        use_cuda=True,
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
        self.gnn = layers.GAT_processor(
            num_layers, num_hidden_features, batch_norm, dropout=dropout, rnf=rnf, granola=granola
        )
        self.predictor = layers.ScorePredictor(
            num_hidden_features * 3, num_hidden_edge_scores
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


class SymGATModel(nn.Module):
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
        rnf=False,
        granola=False,
        use_cuda=True,
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
        self.gnn = layers.SymGAT_processor(
            num_layers, num_hidden_features, batch_norm, dropout=dropout, rnf=rnf, granola=granola
        )
        self.predictor = layers.ScorePredictor(
            num_hidden_features * 3, num_hidden_edge_scores
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
        rnf=False,
        granola=False,
        use_cuda=True,
    ):
        super().__init__()
        self.encoder = layers.NodeEdgeReadsEncoder(
            num_node_features,
            num_edge_features,
            num_intermediate_hidden_features,
            num_hidden_features,
            num_gru_features=32,
        )
        self.gnn = layers.SymGatedGCN_processor(
            num_layers, num_hidden_features, batch_norm, dropout=dropout, rnf=rnf, granola=granola
        )
        self.predictor = layers.ScorePredictor(
            num_hidden_features, num_hidden_edge_scores
        )

    def forward(self, graph, x, e):
        x, e = self.encoder(graph, x, e)
        x, e = self.gnn(graph, x, e)
        scores = self.predictor(graph, x, e)

        return scores


class SymGatedGCNMambaModel(nn.Module):
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
        rnf=False,
        granola=False,
        use_cuda=True,
    ):
        super().__init__()
        self.linear1_node = nn.Linear(
            num_node_features, num_intermediate_hidden_features
        )
        self.linear2_node = nn.Linear(
            num_intermediate_hidden_features, num_hidden_features
        )
        self.mix_mamba_1 = nn.Linear(
            2 * num_hidden_features, num_hidden_features
        )
        self.mix_mamba_2 = nn.Linear(
            num_hidden_features, num_hidden_features
        )
        self.linear1_edge = nn.Linear(
            num_edge_features, num_intermediate_hidden_features
        )
        self.linear2_edge = nn.Linear(
            num_intermediate_hidden_features, num_hidden_features
        )
        self.gnn = layers.SymGatedGCN_processor(
            num_layers, num_hidden_features, batch_norm, dropout=dropout, rnf=rnf, granola=granola
        )
        self.use_cuda = use_cuda
        self.mamba_config = MambaConfig(d_model=4, n_layers=2, d_state=16, use_cuda=use_cuda)
        self.mamba = MambaPytorch(self.mamba_config) # This module uses roughly 3 * expand * d_model^2 parameters
        self.linear_base = nn.Linear(4, num_hidden_features)
        self.predictor = layers.ScorePredictor(
            num_hidden_features * 3, num_hidden_edge_scores
        )

    def forward(self, graph, x, e):
        x = self.linear1_node(x)
        x = torch.relu(x)
        x = self.linear2_node(x)

        e = self.linear1_edge(e)
        e = torch.relu(e)
        e = self.linear2_edge(e)

        idx = graph.ndata["read_length"] - 1  # Subtract 1 to get last index from length
        node_idx = torch.arange(graph.num_nodes())
        reads = graph.ndata["read_data"]
        x2 = self.mamba(reads)[node_idx, idx, :]
        x2 = self.linear_base(x2)
        x = torch.concat([x, x2], dim=-1)
        x = self.mix_mamba_1(x)
        x = torch.relu(x)
        x = self.mix_mamba_2(x)

        x, e = self.gnn(graph, x, e)

        scores = self.predictor(graph, x, e)
        return scores


class SymGatedGCNMambaOnlyModel(nn.Module):
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
        rnf=False,
        granola=False,
        use_cuda=True,
    ):
        super().__init__()
        self.linear1_node = nn.Linear(
            num_node_features, num_intermediate_hidden_features
        )
        self.linear2_node = nn.Linear(
            num_intermediate_hidden_features, num_hidden_features
        )
        self.mix_mamba_1 = nn.Linear(
            2 * num_hidden_features, num_hidden_features
        )
        self.mix_mamba_2 = nn.Linear(
            num_hidden_features, num_hidden_features
        )
        self.linear1_edge = nn.Linear(
            num_hidden_features * 4, num_hidden_features
        )
        self.linear2_edge = nn.Linear(
            num_hidden_features, num_hidden_features
        )
        self.gnn = layers.SymGatedGCN_processor(
            num_layers, num_hidden_features, batch_norm, dropout=dropout, rnf=rnf, granola=granola
        )
        self.use_cuda = use_cuda
        self.mamba_config = MambaConfig(d_model=4, n_layers=2, d_state=16, use_cuda=use_cuda)
        self.mamba = MambaPytorch(self.mamba_config) # This module uses roughly 3 * expand * d_model^2 parameters
        self.linear_base = nn.Linear(4, num_hidden_features)
        self.predictor = layers.ScorePredictor(
            num_hidden_features * 3, num_hidden_edge_scores
        )

    def forward(self, graph, x, e):
        x = self.linear1_node(x)
        x = torch.relu(x)
        x = self.linear2_node(x)

        idx = graph.ndata["read_length"] - 1  # Subtract 1 to get last index from length
        node_idx = torch.arange(graph.num_nodes())
        reads = graph.ndata["read_data"]
        x2 = self.mamba(reads)[node_idx, idx]
        x2 = self.linear_base(x2)

        def apply_edges(edges):
            data = torch.cat((edges.src["x"], edges.dst["x"], edges.src["x2"], edges.dst["x2"]), dim=1)
            h = self.linear1_edge(data)
            h = torch.relu(h)
            e = torch.relu(self.linear2_edge(h))
            return {"e": e}

        e = None

        with graph.local_scope():
            graph.ndata["x"] = x
            graph.ndata["x2"] = x2
            graph.apply_edges(apply_edges)
            e = graph.edata["e"]

        x = torch.concat([x, x2], dim=-1)
        x = self.mix_mamba_1(x)
        x = torch.relu(x)
        x = self.mix_mamba_2(x)

        x, e = self.gnn(graph, x, e)
        scores = self.predictor(graph, x, e)
        return scores


class SymGatedGCNRandomEdgeModel(nn.Module):
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
        rnf=False,
        granola=False,
        use_cuda=True,
    ):
        super().__init__()
        self.linear1_node = nn.Linear(
            num_node_features, num_intermediate_hidden_features
        )
        self.linear2_node = nn.Linear(
            num_intermediate_hidden_features, num_hidden_features
        )
        self.linear1_edge = nn.Linear(
            num_hidden_features * 4, num_hidden_features
        )
        self.linear2_edge = nn.Linear(
            num_hidden_features, num_hidden_features
        )
        self.gnn = layers.SymGatedGCN_processor(
            num_layers, num_hidden_features, batch_norm, dropout=dropout, rnf=rnf, granola=granola
        )
        self.use_cuda = use_cuda
        self.predictor = layers.ScorePredictor(
            num_hidden_features * 3, num_hidden_edge_scores
        )

    def forward(self, graph, x, e):
        x = self.linear1_node(x)
        x = torch.relu(x)
        x = self.linear2_node(x)

        x2 = torch.randn_like(x)

        def apply_edges(edges):
            data = torch.cat((edges.src["x"], edges.dst["x"], edges.src["x2"], edges.dst["x2"]), dim=1)
            h = self.linear1_edge(data)
            h = torch.relu(h)
            e = torch.relu(self.linear2_edge(h))
            return {"e": e}

        e = None

        with graph.local_scope():
            graph.ndata["x"] = x
            graph.ndata["x2"] = x2
            graph.apply_edges(apply_edges)
            e = graph.edata["e"]

        x, e = self.gnn(graph, x, e)
        scores = self.predictor(graph, x, e)
        return scores


class ModelType(Enum):
    SymGatedGCN = SymGatedGCNModel
    GAT = GATModel
    SymGAT = SymGATModel
    SymGatedGCNReads = SymGatedGCNWithReadsModel
    SymGatedGCNMamba = SymGatedGCNMambaModel
    SymGatedGCNMambaOnly = SymGatedGCNMambaOnlyModel
    SymGatedGCNRandomEdge = SymGatedGCNRandomEdgeModel
