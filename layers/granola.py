import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class MPNNLayer(nn.Module):
    def __init__(self, in_channels, rnf=True):
        super(MPNNLayer, self).__init__()
        self.rnf = rnf
        if not self.rnf:
            self.message_fn = nn.Sequential(
                nn.Linear(in_channels * 3, in_channels),
                nn.ReLU(),
                nn.Linear(in_channels, in_channels)
            )
        else:
            self.message_fn = nn.Sequential(
                nn.Linear(in_channels * 5, in_channels),
                nn.ReLU(),
                nn.Linear(in_channels, in_channels)
            )

        self.update_fn = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )

    def message(self, edges):
        msg_input = torch.cat([edges.src['h'], edges.data['e'], edges.dst['h']], dim=1)
        return {'m': self.message_fn(msg_input)}

    def forward(self, g, h, e):
        with g.local_scope():
            if not self.rnf:
                g.ndata['h'] = h
            else:
                rnf = torch.randn_like(h)
                g.ndata['h'] = torch.concat([h, rnf], dim=-1)
            g.edata['e'] = e
            g.update_all(self.message, fn.sum('m', 'm_sum'))
            m_sum = g.ndata['m_sum']
            h_new = self.update_fn(torch.cat([m_sum, h], dim=-1))
            # Note that the edge features are returned unmodified
            return h_new, e


class Granola(nn.Module):
    def __init__(self, in_channels, rnf=True):
        super(Granola, self).__init__()
        self.mpnn_layers = nn.ModuleList(
            [ MPNNLayer(in_channels, rnf) for _ in range(2) ] # A shallow normalization network
        )
        self.gamma_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )
        self.beta_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )

    def forward(self, g, h, e):
        for mpnn_layer in self.mpnn_layers:
            h, e = mpnn_layer(g, h, e)
        z, _ = h, e # Ignore the edge feature output as those are unmodified anyway
        mu = torch.mean(h, dim=-1, keepdim=True)
        sigma = torch.std(h, dim=-1, keepdim=True)
        gamma = self.gamma_fn(z)
        beta = self.beta_fn(z)

        return gamma * ((h - mu) / sigma) + beta, e
