from dataclasses import asdict

import dgl
import lightning as L
import torch
from torch import nn
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dgl import load_graphs
from torch.utils import data
import random
import os
import lightning as L
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import WandbLogger
import time

import utils
from models import ModelType
from pydantic import BaseModel
from dataset import Dataset


class TrainingConfig(BaseModel):
    model_type: ModelType = ModelType.SymGatedGCN
    num_node_features: int = 2
    num_edge_features: int = 2
    num_intermediate_hidden_features: int = 16
    num_hidden_features: int = 64
    num_layers: int = 8
    num_hidden_edge_scores: int = 64
    batch_norm: bool = True
    use_cuda: bool = True # Setting use_cuda to False uses an alternative PyTorch-based parallel scan implementation, rather than CUDA

    # Training hyperparameters
    seed: int = 42
    training_chromosomes: list[int] = [21]
    validation_chromosomes: list[int] = [19]

def preprocess_graph(g):
    g = g.int()
    g.ndata['x'] = torch.ones(g.num_nodes(), 1)
    ol_len = g.edata['overlap_length'].float()
    ol_sim = g.edata['overlap_similarity']
    ol_len = (ol_len - ol_len.mean()) / ol_len.std()
    g.edata['e'] = torch.cat((ol_len.unsqueeze(-1), ol_sim.unsqueeze(-1)), dim=1)
    return g


def add_positional_encoding(g):
    g.ndata['in_deg'] = g.in_degrees().float()
    g.ndata['out_deg'] = g.out_degrees().float()
    return g


def mask_graph(g, fraction, device):
    keep_node_idx = torch.rand(g.num_nodes(), device=device) < fraction
    sub_g = dgl.node_subgraph(g, keep_node_idx, store_ids=True)
    return sub_g


def mask_graph_strandwise(g, fraction, device):
    keep_node_idx_half = torch.rand(g.num_nodes() // 2, device=device) < fraction
    keep_node_idx = torch.empty(keep_node_idx_half.size(0) * 2, dtype=keep_node_idx_half.dtype)
    keep_node_idx[0::2] = keep_node_idx_half
    keep_node_idx[1::2] = keep_node_idx_half
    sub_g = dgl.node_subgraph(g, keep_node_idx, store_ids=True)
    print(f'Masking fraction: {fraction}')
    print(f'Original graph: N={g.num_nodes()}, E={g.num_edges()}')
    print(f'Subsampled graph: N={sub_g.num_nodes()}, E={sub_g.num_edges()}')
    return sub_g


def symmetry_loss(org_scores, rev_scores, labels, pos_weight=1.0, alpha=1.0):
    BCE = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    BCE_org = BCE(org_scores, labels)
    BCE_rev = BCE(rev_scores, labels)
    abs_diff = torch.abs(org_scores - rev_scores)
    loss = (BCE_org + BCE_rev + alpha * abs_diff)
    loss = loss.mean()
    return loss


class SubgraphDataset(data.Dataset):
    def mask_graph_strandwise(self, g, fraction):
        keep_node_idx_half = torch.rand(g.num_nodes() // 2) < fraction
        keep_node_idx = torch.empty(keep_node_idx_half.size(0) * 2, dtype=keep_node_idx_half.dtype)
        keep_node_idx[0::2] = keep_node_idx_half
        keep_node_idx[1::2] = keep_node_idx_half
        sub_g = dgl.node_subgraph(g, keep_node_idx, store_ids=True)
        print(f'Masking fraction: {fraction}')
        print(f'Original graph: N={g.num_nodes()}, E={g.num_edges()}')
        print(f'Subsampled graph: N={sub_g.num_nodes()}, E={sub_g.num_edges()}')
        return sub_g

    def repartition(self):
        self.num_subgraphs_accessed = 0
        self.subgraphs = []

        for graph in self.graphs:
            fraction = random.randint(self.mask_frac_low, self.mask_frac_high) / 100  # Fraction of nodes to be left in the graph (.85 -> ~30x, 1.0 -> 60x)
            graph = self.mask_graph_strandwise(graph, fraction)

            # Number of clusters dependant on graph size!
            num_nodes_per_cluster_min = int(self.num_nodes_per_cluster * self.npc_lower_bound)
            num_nodes_per_cluster_max = int(self.num_nodes_per_cluster * self.npc_upper_bound) + 1
            num_nodes_for_g = torch.LongTensor(1).random_(num_nodes_per_cluster_min, num_nodes_per_cluster_max).item()
            num_clusters = graph.num_nodes() // num_nodes_for_g + 1

            graph = graph.long()
            d = dgl.metis_partition(graph, num_clusters, extra_cached_hops=1)
            sub_gs = list(d.values())
            transformed_sub_gs = []

            for sub_g in sub_gs:
                e = graph.edata['e'][sub_g.edata[dgl.EID]]
                pe_in = graph.ndata['in_deg'][sub_g.ndata[dgl.NID]].unsqueeze(1)
                pe_in = (pe_in - pe_in.mean()) / pe_in.std()
                pe_out = graph.ndata['out_deg'][sub_g.ndata[dgl.NID]].unsqueeze(1)
                pe_out = (pe_out - pe_out.mean()) / pe_out.std()
                pe = torch.cat((pe_in, pe_out), dim=1)
                labels = graph.edata['y'][sub_g.edata[dgl.EID]]

                rev_sub_g = dgl.reverse(sub_g, True, True)
                rev_e = graph.edata['e'][rev_sub_g.edata[dgl.EID]]
                pe_out = graph.ndata['in_deg'][rev_sub_g.ndata[dgl.NID]].unsqueeze(1)
                pe_out = (pe_out - pe_out.mean()) / pe_out.std()
                pe_in = graph.ndata['out_deg'][rev_sub_g.ndata[dgl.NID]].unsqueeze(1)  # Reversed edges, in/out-deg also reversed
                pe_in = (pe_in - pe_in.mean()) / pe_in.std()
                rev_pe = torch.cat((pe_in, pe_out), dim=1)

                transformed_sub_gs.append((sub_g, pe, e, rev_sub_g, rev_pe, rev_e, labels))

            self.subgraphs += transformed_sub_gs

        if self.shuffle:
            random.shuffle(self.subgraphs)

    def __init__(self, mask_frac_low=80, mask_frac_high=100, num_nodes_per_cluster=2000, npc_lower_bound=1, npc_upper_bound=1, shuffle=True):
        self.mask_frac_low = mask_frac_low
        self.mask_frac_high = mask_frac_high
        self.num_nodes_per_cluster = num_nodes_per_cluster
        self.npc_lower_bound = npc_lower_bound
        self.npc_upper_bound = npc_upper_bound
        self.shuffle = shuffle

        (loaded_graph,), _ = dgl.load_graphs(os.path.join('chm13htert-data/chr19/', 'hifiasm/processed/0.dgl'))
        loaded_graph = preprocess_graph(loaded_graph)
        loaded_graph = add_positional_encoding(loaded_graph)
        self.graphs = [loaded_graph]
        self.subgraphs = []

        self.pos_to_neg_ratio = sum([((torch.round(g.edata['y'])==1).sum() / (torch.round(g.edata['y'])==0).sum()).item() for g in self.graphs]) / len(self.graphs)

        self.num_subgraphs_accessed = 0
        self.repartition()

    def __len__(self):
        return len(self.subgraphs)

    def __getitem__(self, idx):
        if self.num_subgraphs_accessed > len(self.subgraphs):
            self.repartition()
        self.num_subgraphs_accessed += 1
        return self.subgraphs[idx]


class Model(L.LightningModule):
    def __init__(self, training_config: TrainingConfig):
        super().__init__()
        self.save_hyperparameters(training_config.model_dump())
        self.model = training_config.model_type.value(
            training_config.num_node_features,
            training_config.num_edge_features,
            training_config.num_intermediate_hidden_features,
            training_config.num_hidden_features,
            training_config.num_layers,
            training_config.num_hidden_edge_scores,
            training_config.batch_norm,
            training_config.use_cuda,
        )


        self.pos_to_neg_ratio = ds_train.pos_to_neg_ratio

    def training_step(self, batch, batch_idx):
        sub_g, pe, e, rev_sub_g, rev_pe, rev_e, labels = batch
        # sub_g, pe, e, rev_sub_g, rev_pe, rev_e, labels = sub_g.to('cuda'), pe.to('cuda'), e.to('cuda'), rev_sub_g.to('cuda'), rev_pe.to('cuda'), rev_e.to('cuda'), labels.to('cuda')
        org_scores = self.model(sub_g, pe, e).squeeze(-1)

        rev_scores = self.model(rev_sub_g, rev_pe, rev_e).squeeze(-1)

        pos_weight = torch.tensor([1 / self.pos_to_neg_ratio], device=org_scores.device)
        loss = symmetry_loss(org_scores, rev_scores, labels, pos_weight, alpha=0.1)
        edge_predictions = org_scores
        edge_labels = labels

        TP, TN, FP, FN = utils.calculate_tfpn(edge_predictions, edge_labels)
        acc, precision, recall, f1 =  utils.calculate_metrics(TP, TN, FP, FN)
        acc_inv, precision_inv, recall_inv, f1_inv =  utils.calculate_metrics_inverse(TP, TN, FP, FN)

        try:
            fp_rate = FP / (FP + TN)
        except ZeroDivisionError:
            fp_rate = 0.0
        try:
            fn_rate = FN / (FN + TP)
        except ZeroDivisionError:
            fn_rate = 0.0

        # Append results of a single mini-batch / METIS partition
        self.log("running_loss", loss.item(), prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
        self.log("running_fp_rate", fp_rate, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
        self.log("running_fn_rate", fn_rate, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
        self.log("running_acc", acc, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
        self.log("running_precision", precision, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
        self.log("running_recall", recall, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
        self.log("running_f1", f1, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)

        # Inverse metrics because F1 and them are not good for dataset with mostly positive labels
        self.log("train_acc_inv_epoch", acc_inv, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
        self.log("train_precision_inv_epoch", precision_inv, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
        self.log("train_recall_inv_epoch", recall_inv, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
        self.log("train_f1_inv_epoch", f1_inv, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.95, patience=2, verbose=True)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "running_loss",
                    "interval": "epoch",
                    "frequency": 1,
                    # If "monitor" references validation metrics, then "frequency" should be set to a
                    # multiple of "trainer.check_val_every_n_epoch".
                },
            }


ds_train = SubgraphDataset()
cfg = TrainingConfig()
model = Model(cfg)

seed_everything(cfg.seed, workers=True)

wandb_logger = WandbLogger(project="new-genome-assembly", name=(cfg.model_type.value.__name__ + "_seed=" + str(cfg.seed) + "_time=" + str(time.time())), resume="never")

trainer = L.Trainer(max_epochs=250, log_every_n_steps=1, deterministic=True, logger=wandb_logger)
trainer.fit(model=model, train_dataloaders=ds_train)
