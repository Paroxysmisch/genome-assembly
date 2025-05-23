from dataclasses import asdict

import dgl
import lightning as L
import torch
from torch import nn
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dgl import load_graphs

import utils
from models import ModelType
from pydantic import BaseModel
from dataset import Dataset


class TrainingConfig(BaseModel):
    model_type: ModelType = ModelType.SymGatedGCN
    num_node_features: int = 2
    num_edge_features: int = 1
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


def symmetry_loss(org_scores, rev_scores, labels, pos_weight=1.0, alpha=1.0):
    BCE = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight), reduction="none"
    )
    BCE_org = BCE(org_scores, labels)
    BCE_rev = BCE(rev_scores, labels)
    abs_diff = torch.abs(org_scores - rev_scores)
    loss = BCE_org + BCE_rev + alpha * abs_diff
    loss = loss.mean()
    return loss


# def calculate_node_and_edge_features(graph, sub_g):
#     # graph.edata["overlap_length"].to(sub_g.device)
#     ol_len = graph.edata["overlap_length"][sub_g.edata[dgl.EID]].float()
#     ol_len = (ol_len - ol_len.mean()) / ol_len.std()
#     e = ol_len.unsqueeze(-1)
#
#     pe_in = graph.in_degrees()[sub_g.ndata[dgl.NID]].float().unsqueeze(1)
#     pe_in = (pe_in - pe_in.mean()) / pe_in.std()
#     pe_out = graph.out_degrees()[sub_g.ndata[dgl.NID]].float().unsqueeze(1)
#     pe_out = (pe_out - pe_out.mean()) / pe_out.std()
#     pe = torch.cat((pe_in, pe_out), dim=1)
#
#     return pe, e


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

        # Compute the training and validation pos_weight
        dataset = Dataset.CHM13htert
        raw_dir = dataset.value + "/raw/"

        training_zeros = 0
        training_ones = 0
        total_training_edges = 0
        for chromosome in training_config.training_chromosomes:
            print(f"Calculating pos_weight for chromosome {chromosome}")
            graph_path = raw_dir + "chr" + str(chromosome) + ".dgl"
            (graph,), _ = load_graphs(graph_path)
            training_zeros += (1 - graph.edata['y']).count_nonzero()
            training_ones += graph.edata['y'].count_nonzero()
            total_training_edges += len(graph.edata['y'])
        self.training_pos_weight = training_zeros / training_ones

        # graph_path = raw_dir + "chr" + str(training_config.training_chromosomes[0]) + ".dgl"
        # (graph,), _ = load_graphs(graph_path)
        # self.training_graph = graph

        validation_zeros = 0
        validation_ones = 0
        total_validation_edges = 0
        for chromosome in training_config.validation_chromosomes:
            print(f"Calculating pos_weight for chromosome {chromosome}")
            graph_path = raw_dir + "chr" + str(chromosome) + ".dgl"
            (graph,), _ = load_graphs(graph_path)
            validation_zeros += (1 - graph.edata['y']).count_nonzero()
            validation_ones += graph.edata['y'].count_nonzero()
            total_validation_edges += len(graph.edata['y'])
        self.validation_pos_weight = validation_zeros / validation_ones

        # graph_path = raw_dir + "chr" + str(training_config.validation_chromosomes[0]) + ".dgl"
        # (graph,), _ = load_graphs(graph_path)
        # self.validation_graph = graph

        print(f"Computed training_pos_weight as {self.training_pos_weight}, and validation_pos_weight as {self.validation_pos_weight}")


    def training_step(self, batch, batch_idx):
        sub_g = batch

        # pe, e = calculate_node_and_edge_features(self.training_graph, sub_g)
        pe, e = sub_g.ndata['pe'], sub_g.edata['e']
        org_scores = self.model(sub_g, pe, e).squeeze(-1)

        labels = sub_g.edata["y"]

        sub_g = dgl.reverse(sub_g, True, True)
        # pe, e = calculate_node_and_edge_features(self.training_graph, sub_g)
        pe, e = sub_g.ndata['pe_rev'], sub_g.edata['e']
        rev_scores = self.model(sub_g, pe, e).squeeze(-1)

        loss = symmetry_loss(org_scores, rev_scores, labels, self.training_pos_weight, alpha=0.1)
        TP, TN, FP, FN = utils.calculate_tfpn(
            edge_predictions=org_scores, edge_labels=labels
        )
        accuracy, precision, recall, f1 = utils.calculate_metrics(TP, TN, FP, FN)
        try:
            fp_rate = FP / (FP + TN)
        except ZeroDivisionError:
            fp_rate = 0.0
        try:
            fn_rate = FN / (FN + TP)
        except ZeroDivisionError:
            fn_rate = 0.0
        # if (labels == 0).any():
        #     print(org_scores[labels == 0])
        #     print(rev_scores[labels == 0])

        self.log("train_loss", loss, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)

        self.log("train_accuracy", accuracy, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
        self.log("train_precision", precision, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
        self.log("train_recall", recall, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
        self.log("train_f1", f1, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)

        self.log("train_TP", TP, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
        self.log("train_TN", TN, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
        self.log("train_FP", FP, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
        self.log("train_FN", FN, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
        self.log("train_FP_rate", fp_rate, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
        self.log("train_FN_rate", fn_rate, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)

        return loss

    # def validation_step(self, batch, batch_idx):
    #     sub_g = batch
    #
    #     # pe, e = calculate_node_and_edge_features(self.validation_graph, sub_g)
    #     pe, e = sub_g.ndata['pe'], sub_g.edata['e']
    #     org_scores = self.model(sub_g, pe, e).squeeze(-1)
    #
    #     labels = sub_g.edata["y"]
    #
    #     sub_g = dgl.reverse(sub_g, True, True)
    #     # pe, e = calculate_node_and_edge_features(self.validation_graph, sub_g)
    #     pe, e = sub_g.ndata['pe'], sub_g.edata['e']
    #     rev_scores = self.model(sub_g, pe, e).squeeze(-1)
    #
    #     loss = symmetry_loss(org_scores, rev_scores, labels, self.validation_pos_weight, alpha=0.1)
    #     TP, TN, FP, FN = utils.calculate_tfpn(
    #         edge_predictions=org_scores, edge_labels=labels
    #     )
    #     accuracy, precision, recall, f1 = utils.calculate_metrics(TP, TN, FP, FN)
    #
    #     self.log("validation_loss", loss, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
    #
    #     self.log("validation_accuracy", accuracy, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
    #     self.log("validation_precision", precision, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
    #     self.log("validation_recall", recall, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
    #     self.log("validation_f1", f1, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
    #
    #     self.log("validation_TP", TP, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
    #     self.log("validation_TN", TN, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
    #     self.log("validation_FP", FP, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
    #     self.log("validation_FN", FN, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
    #
    #     return loss

    def forward(self, batch):
        sub_g = batch
        # pe, e = calculate_node_and_edge_features(sub_g)
        pe, e = sub_g.ndata['pe'], sub_g.edata['e']

        return self.model(sub_g, pe, e).squeeze(-1)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.95, patience=2, verbose=True)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train_loss",
                    "interval": "epoch",
                    "frequency": 1,
                    # If "monitor" references validation metrics, then "frequency" should be set to a
                    # multiple of "trainer.check_val_every_n_epoch".
                },
            }
