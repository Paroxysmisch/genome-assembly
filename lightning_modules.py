import dgl
import lightning as L
import torch
from torch import nn
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import SymGatedGCNModel


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


class Model(L.LightningModule):
    def __init__(self, pos_weight):
        super().__init__()
        self.save_hyperparameters()
        self.model = SymGatedGCNModel(
            node_features=2,
            edge_features=1,
            hidden_features=64,
            hidden_edge_features=16,
            num_layers=8,
            hidden_edge_scores=64,
            batch_norm=True,
        )
        self.pos_weight = pos_weight

    def training_step(self, batch, batch_idx):
        sub_g = batch
        ol_len = sub_g.edata['overlap_length'].float()
        ol_len = (ol_len - ol_len.mean()) / ol_len.std()
        e = ol_len.unsqueeze(-1)
        pe_in = sub_g.in_degrees().float().unsqueeze(1)
        pe_in = (pe_in - pe_in.mean()) / pe_in.std()
        pe_out = sub_g.out_degrees().float().unsqueeze(1)
        pe_out = (pe_out - pe_out.mean()) / pe_out.std()
        pe = torch.cat((pe_in, pe_out), dim=1)
        org_scores = self.model(sub_g, pe, e).squeeze(-1)
        labels = sub_g.edata["y"]

        sub_g = dgl.reverse(sub_g, True, True)
        ol_len = sub_g.edata['overlap_length'].float()
        ol_len = (ol_len - ol_len.mean()) / ol_len.std()
        e = ol_len.unsqueeze(-1)
        pe_in = sub_g.in_degrees().float().unsqueeze(1)
        pe_in = (pe_in - pe_in.mean()) / pe_in.std()
        pe_out = sub_g.out_degrees().float().unsqueeze(1)
        pe_out = (pe_out - pe_out.mean()) / pe_out.std()
        pe = torch.cat((pe_in, pe_out), dim=1)
        rev_scores = self.model(sub_g, pe, e).squeeze(-1)

        loss = symmetry_loss(org_scores, rev_scores, labels, self.pos_weight, alpha=0.1)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def forward(self, batch):
        sub_g = batch
        ol_len = sub_g.edata['overlap_length'].float()
        ol_len = (ol_len - ol_len.mean()) / ol_len.std()
        e = ol_len.unsqueeze(-1)
        pe_in = sub_g.in_degrees().float().unsqueeze(1)
        pe_in = (pe_in - pe_in.mean()) / pe_in.std()
        pe_out = sub_g.out_degrees().float().unsqueeze(1)
        pe_out = (pe_out - pe_out.mean()) / pe_out.std()
        pe = torch.cat((pe_in, pe_out), dim=1)
        return self.model(sub_g, pe, e).squeeze(-1)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.95, patience=2)
        return optimizer
        # return [optimizer], [scheduler]
