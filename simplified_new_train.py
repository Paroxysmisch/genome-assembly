import argparse
from datetime import datetime
import os
import random
import statistics
import pickle
import gc

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import dgl
import wandb
from torch.utils import data
from pydantic import BaseModel
from models import ModelType

import models
import utils



# def get_hyperparameters():
#     return {
#         
#         'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
#         # 'device': 'cpu',
#         'seed': 1,
#         'wandb_mode': 'online',  # switch between 'online' and 'disabled'
#         'wandb_project': 'new-genome-assembly',
#
#         'chr_overfit': 0,
#         'plot_nga50_during_training': False,
#         'eval_frequency': 20, 
#
#         # Data
#         'use_similarities': True,
#
#         # Model
#         'dim_latent': 64,
#         'num_gnn_layers': 8,
#         'node_features': 2,
#         'edge_features': 2,  # Put 2 if you use similarities, 1 otherwise
#         'hidden_edge_features': 16,
#         'hidden_edge_scores': 64,
#         'nb_pos_enc': 0,
#         'type_pos_enc': 'none',
#         'batch_norm': True,
#         # 'dropout': 0.08,
#
#         # Training
#         'num_epochs': 250,
#         'lr': 1e-4,
#         'use_symmetry_loss': True,
#         'alpha': 0.1,
#         'num_parts_metis_train': 200,
#         'num_parts_metis_eval': 200,
#         'num_nodes_per_cluster': 2000,  # 2000 = max 10GB GPU memory for d=128, L=8
#         'npc_lower_bound': 1,  # 0.8
#         'npc_upper_bound': 1,  # 1.2
#         'k_extra_hops': 1,
#         'patience': 2,
#         'decay': 0.95,
#         'masking': True,
#         'mask_frac_low': 80,   # ~ 25x
#         'mask_frac_high': 100, # ~ 60x
#
#         # Decoding
#         'strategy': 'greedy',
#         'num_decoding_paths': 100,
#         'decode_with_labels': False,
#         'load_checkpoint': True,
#         'num_threads': 32,
#         'B': 1,
#         'len_threshold': 10,
#     }


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

def save_checkpoint(epoch, model, optimizer, loss_train, loss_valid, out, ckpt_path):
    checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'loss_train': loss_train,
            'loss_valid': loss_valid,
    }
    torch.save(checkpoint, ckpt_path)


def load_checkpoint(out, model, optimizer):
    ckpt_path = f'checkpoints/{out}.pt'
    checkpoint = torch.load(ckpt_path)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    loss_train = checkpoint['loss_train']
    loss_valid = checkpoint['loss_valid']
    return epoch, model, optimizer, loss_train, loss_valid


def view_model_param(model):
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    return total_param


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

        for graph, graph_reads in zip(self.graphs, self.reads):
            if self.is_train:
                fraction = random.uniform(self.mask_frac_low, self.mask_frac_high)  # Fraction of nodes to be left in the graph (.85 -> ~30x, 1.0 -> 60x)
                graph = self.mask_graph_strandwise(graph, fraction)

                # Number of clusters dependant on graph size!
                num_nodes_per_cluster_min = int(self.num_nodes_per_cluster * self.npc_lower_bound)
                num_nodes_per_cluster_max = int(self.num_nodes_per_cluster * self.npc_upper_bound) + 1
                num_nodes_for_g = torch.LongTensor(1).random_(num_nodes_per_cluster_min, num_nodes_per_cluster_max).item()
                num_clusters = graph.num_nodes() // num_nodes_for_g + 1

                graph = graph.long()
                d = dgl.metis_partition(graph, num_clusters, extra_cached_hops=1)
            else:
                num_clusters = graph.num_nodes() // self.num_nodes_per_cluster + 1

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

                # Add the read data
                sub_g.ndata['read_length'] = torch.min(torch.tensor(self.max_length), graph.ndata['read_length'][sub_g.ndata[dgl.NID]])
                rev_sub_g.ndata['read_length'] = torch.min(torch.tensor(self.max_length), graph.ndata['read_length'][rev_sub_g.ndata[dgl.NID]])
                sub_g.ndata['read_data'] = graph_reads[sub_g.ndata[dgl.NID]]
                rev_sub_g.ndata['read_data'] = graph_reads[rev_sub_g.ndata[dgl.NID]]
                transformed_sub_gs.append((sub_g, pe, e, rev_sub_g, rev_pe, rev_e, labels))

            self.subgraphs += transformed_sub_gs

        if self.shuffle:
            random.shuffle(self.subgraphs)

    def __init__(self, cfg, is_train=True, mask_frac_low=80, mask_frac_high=100, num_nodes_per_cluster=2000, npc_lower_bound=1, npc_upper_bound=1, shuffle=True):
        self.mask_frac_low = mask_frac_low
        self.mask_frac_high = mask_frac_high
        self.num_nodes_per_cluster = num_nodes_per_cluster
        self.npc_lower_bound = npc_lower_bound
        self.npc_upper_bound = npc_upper_bound
        self.shuffle = shuffle
        self.is_train = is_train

        self.graphs = []
        self.subgraphs = []
        if self.is_train:
            for chromosome in cfg.training_chromosomes:
                (loaded_graph,), _ = dgl.load_graphs(os.path.join(cfg.data_dir, f'chr{str(chromosome)}/', 'hifiasm/processed/0.dgl'))
                loaded_graph = preprocess_graph(loaded_graph)
                loaded_graph = add_positional_encoding(loaded_graph)
                self.graphs.append(loaded_graph)
        else:
            for chromosome in cfg.validation_chromosomes:
                (loaded_graph,), _ = dgl.load_graphs(os.path.join(cfg.data_dir, f'chr{str(chromosome)}/', 'hifiasm/processed/0.dgl'))
                loaded_graph = preprocess_graph(loaded_graph)
                loaded_graph = add_positional_encoding(loaded_graph)
                self.graphs.append(loaded_graph)

        self.pos_to_neg_ratio = sum([((torch.round(g.edata['y'])==1).sum() / (torch.round(g.edata['y'])==0).sum()).item() for g in self.graphs]) / len(self.graphs)

        self.max_length = 6000
        self.reads = []
        chromosomes = cfg.training_chromosomes if self.is_train else cfg.validation_chromosomes
        for chromosome in chromosomes:
            reads_path = os.path.join(cfg.data_dir, f'chr{str(chromosome)}/', 'hifiasm/info/0_reads.pkl')
            with open(reads_path, "rb") as f:
                reads_dict = pickle.load(f)

                # Pad reads with 'N' to max_length
                padded_reads = [read.ljust(self.max_length, 'N')[:self.max_length] for read in reads_dict.values()]

                # Create ASCII mapping
                mapping = torch.full((128,), -1, dtype=torch.long)  # default -1 for unknowns
                mapping[ord('A')] = 0
                mapping[ord('C')] = 1
                mapping[ord('G')] = 2
                mapping[ord('T')] = 3
                mapping[ord('N')] = 0  # Special padding token

                # Convert characters to integers
                reads_ascii = torch.tensor([list(map(ord, read)) for read in padded_reads])
                reads_int = mapping[reads_ascii]

                # One-hot encode
                one_hot = torch.nn.functional.one_hot(reads_int, num_classes=4)  # 4 because N takes the same representation as A---fine as we truncate the Mamba output according to read_length

                self.reads.append(one_hot.float())  # shape: (num_reads, max_length, 4)

        self.num_subgraphs_accessed = 0
        self.repartition()

    def __len__(self):
        return len(self.subgraphs)

    def __getitem__(self, idx):
        if self.num_subgraphs_accessed > len(self.subgraphs) and self.is_train:
            self.repartition()
        self.num_subgraphs_accessed += 1
        return self.subgraphs[idx]


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
    data_dir: str = "chm13htert-data/"
    training_chromosomes: list[int] = [21]
    validation_chromosomes: list[int] = [19]
    seed: int = 42
    num_epochs: int = 250
    patience: int = 10
    learning_rate: float = 1e-4
    device: str = "cuda:0"
    decay: float = 0.75
    alpha: float = 0.1
    mask_frac_low: float = 0.8
    mask_frac_high: float = 1.0
    num_nodes_per_cluster: int = 2000

def train(train_path, valid_path, out, assembler, overfit=False, dropout=None, seed=None, resume=False, finetune=False, ft_model=None):
    cfg = TrainingConfig()
    seed = cfg.seed
    num_epochs = cfg.num_epochs
    num_gnn_layers = cfg.num_layers
    hidden_features = cfg.num_hidden_features
    patience = cfg.patience
    lr = cfg.learning_rate
    device = cfg.device
    batch_norm = cfg.batch_norm
    node_features = cfg.num_node_features
    edge_features = cfg.num_edge_features
    hidden_edge_features = cfg.num_intermediate_hidden_features
    hidden_edge_scores = cfg.num_hidden_edge_scores
    decay = cfg.decay
    wandb_mode = "online"
    wandb_project = "new-genome-assembly"
    alpha = cfg.alpha
    mask_frac_low = cfg.mask_frac_low
    mask_frac_high = cfg.mask_frac_high
    num_nodes_per_cluster = cfg.num_nodes_per_cluster

    checkpoints_path = os.path.abspath("artifacts/checkpoints")
    models_path = os.path.abspath("artifacts/models")

    print(f'----- TRAIN -----')
    print(f'\nSaving checkpoints: {checkpoints_path}')
    print(f'Saving models: {models_path}\n')

    print(f'USING SEED: {seed}')

    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    utils.set_seed(seed)

    time_start = datetime.now()
    timestamp = time_start.strftime('%Y-%b-%d-%H-%M-%S')

    if out is None:
        out = timestamp

    # if not overfit:
    #     ds_train = AssemblyGraphDataset(train_path, assembler=assembler)
    #     ds_valid = AssemblyGraphDataset(valid_path, assembler=assembler)
    # else:
    #     ds_train = ds_valid = AssemblyGraphDataset(train_path, assembler=assembler)

    ds_train = SubgraphDataset(cfg, True, mask_frac_low, mask_frac_high, num_nodes_per_cluster)
    ds_valid = SubgraphDataset(cfg, False, mask_frac_low, mask_frac_high, num_nodes_per_cluster)

    pos_to_neg_ratio = ds_train.pos_to_neg_ratio
    validation_pos_to_neg_ratio = ds_valid.pos_to_neg_ratio

    model = cfg.model_type.value(node_features, edge_features, hidden_edge_features, hidden_features, num_gnn_layers, hidden_edge_scores, batch_norm, dropout=dropout)
    model.to(device)
    if not os.path.exists(models_path):
        print(models_path)
        os.makedirs(models_path)

    out = out + f'_seed{seed}'

    model_path = os.path.join(models_path, f'model_{out}.pt')    
    print(f'MODEL PATH: {model_path}')

    ckpt_path = f'{checkpoints_path}/ckpt_{out}.pt'
    print(f'CHECKPOINT PATH: {ckpt_path}')

    print(f'\nNumber of network parameters: {view_model_param(model)}\n')
    print(f'Normalization type : Batch Normalization\n') if batch_norm else print(f'Normalization type : Layer Normalization\n')

    pos_weight = torch.tensor([1 / pos_to_neg_ratio], device=device)
    validation_pos_weight = torch.tensor([1 / validation_pos_to_neg_ratio], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=decay, patience=patience, verbose=True)
    start_epoch = 0

    loss_per_epoch_train, loss_per_epoch_valid = [], []

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    elapsed = utils.timedelta_to_str(datetime.now() - time_start)
    print(f'Loading data done. Elapsed time: {elapsed}')

    try:
        with wandb.init(project=wandb_project, config=cfg.model_dump(), mode=wandb_mode, name=out):
            wandb.watch(model, criterion, log='all', log_freq=1000)

            for epoch in range(start_epoch, num_epochs):
                train_loss_epoch, train_fp_rate_epoch, train_fn_rate_epoch = [], [], []

                print('\n===> TRAINING\n')
                model.train()

                print(f'\n(TRAIN: Epoch = {epoch:3})')

                # Loop over all mini-batch i.e. subgraphs in the collection of graphs
                for batch in ds_train:
                    sub_g, pe, e, rev_sub_g, rev_pe, rev_e, labels = batch
                    sub_g, pe, e, rev_sub_g, rev_pe, rev_e, labels = sub_g.to(device), pe.to(device), e.to(device), rev_sub_g.to(device), rev_pe.to(device), rev_e.to(device), labels.to(device)
                    # Runs the forward pass with autocasting.
                    # with torch.autocast(device_type=cfg.device, dtype=torch.bfloat16):
                    org_scores = model(sub_g, pe, e).squeeze(-1)

                    rev_scores = model(rev_sub_g, rev_pe, rev_e).squeeze(-1)

                    loss = symmetry_loss(org_scores, rev_scores, labels, pos_weight, alpha=alpha)
                    edge_predictions = org_scores
                    edge_labels = labels

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

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
                    wandb.log(
                        {
                            "loss": loss.item(),
                            "fp_rate": fp_rate,
                            "fn_rate": fn_rate,
                            "acc": acc,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "acc_inv": acc_inv,
                            "precision_inv": precision_inv,
                            "recall_inv": recall_inv,
                            "f1_inv": f1_inv,
                            "lr_value": optimizer.param_groups[0]['lr'],
                            "epoch": epoch,
                        }
                    )

                    train_loss_epoch.append(loss.item())
                    train_fp_rate_epoch.append(fp_rate)
                    train_fn_rate_epoch.append(fn_rate)

                    # After finishing the batch, delete the subgraphs
                    del sub_g
                    del rev_sub_g

                    # Run garbage collection to clear any leftover references
                    gc.collect()

                    # Clear CUDA cache (useful after deletion)
                    torch.cuda.empty_cache()

                train_loss_epoch = statistics.mean(train_loss_epoch)
                train_fp_rate_epoch = statistics.mean(train_fp_rate_epoch)
                train_fn_rate_epoch = statistics.mean(train_fn_rate_epoch)
                loss_per_epoch_train.append(train_loss_epoch)

                elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                print(f'\n==> TRAINING (all training graphs): Epoch = {epoch}')
                print(f'Loss: {train_loss_epoch:.4f}, fp_rate(GT=0): {train_fp_rate_epoch:.4f}, fn_rate(GT=1): {train_fn_rate_epoch:.4f}')
                print(f'Elapsed time: {elapsed}\n\n')

                if overfit:
                    if len(loss_per_epoch_valid) == 1 or len(loss_per_epoch_train) > 1 and loss_per_epoch_train[-1] < min(loss_per_epoch_train[:-1]):
                        torch.save(model.state_dict(), model_path)
                        print(f'Epoch {epoch}: Model saved!')
                    save_checkpoint(epoch, model, optimizer, loss_per_epoch_train[-1], 0.0, out, ckpt_path)
                    scheduler.step(train_loss_epoch)

                    continue  # This will entirely skip the validation

                if epoch % 5 == 0:
                    model.eval()

                    validation_loss_epoch, validation_fp_rate_epoch, validation_fn_rate_epoch = [], [], []

                    for batch in ds_valid:
                        sub_g, pe, e, rev_sub_g, rev_pe, rev_e, labels = batch
                        sub_g, pe, e, rev_sub_g, rev_pe, rev_e, labels = sub_g.to(device), pe.to(device), e.to(device), rev_sub_g.to(device), rev_pe.to(device), rev_e.to(device), labels.to(device)
                        # Runs the forward pass with autocasting.
                        with torch.autocast(device_type=cfg.device, dtype=torch.bfloat16):
                            org_scores = model(sub_g, pe, e).squeeze(-1)

                            rev_scores = model(rev_sub_g, rev_pe, rev_e).squeeze(-1)

                            loss = symmetry_loss(org_scores, rev_scores, labels, validation_pos_weight, alpha=alpha)
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
                        wandb.log(
                            {
                                "validation_loss": loss.item(),
                                "validation_fp_rate": fp_rate,
                                "validation_fn_rate": fn_rate,
                                "validation_acc": acc,
                                "validation_precision": precision,
                                "validation_recall": recall,
                                "validation_f1": f1,
                                "validation_acc_inv": acc_inv,
                                "validation_precision_inv": precision_inv,
                                "validation_recall_inv": recall_inv,
                                "validation_f1_inv": f1_inv,
                                "validation_epoch": epoch,
                            }
                        )

                        validation_loss_epoch.append(loss.item())
                        validation_fp_rate_epoch.append(fp_rate)
                        validation_fn_rate_epoch.append(fn_rate)

                        # After finishing the batch, delete the subgraphs
                        del sub_g
                        del rev_sub_g

                        # Run garbage collection to clear any leftover references
                        gc.collect()

                        # Clear CUDA cache (useful after deletion)
                        torch.cuda.empty_cache()

                    validation_loss_epoch = statistics.mean(validation_loss_epoch)
                    validation_fp_rate_epoch = statistics.mean(validation_fp_rate_epoch)
                    validation_fn_rate_epoch = statistics.mean(validation_fn_rate_epoch)
                    loss_per_epoch_valid.append(validation_loss_epoch)

                    print(f'\n==> VALIDATION (all validation graphs): Epoch = {epoch}')
                    print(f'Loss: {validation_loss_epoch:.4f}, fp_rate(GT=0): {validation_fp_rate_epoch:.4f}, fn_rate(GT=1): {validation_fn_rate_epoch:.4f}')

                    if len(loss_per_epoch_valid) == 1 or len(loss_per_epoch_valid) > 1 and loss_per_epoch_valid[-1] < min(loss_per_epoch_valid[:-1]):
                        torch.save(model.state_dict(), model_path)
                        print(f'Epoch {epoch}: Model saved!')
                    save_checkpoint(epoch, model, optimizer, loss_per_epoch_valid[-1], 0.0, out, ckpt_path)

    except KeyboardInterrupt:
        torch.cuda.empty_cache()
        print("Keyboard Interrupt...")
        print("Exiting...")

    finally:
        torch.cuda.empty_cache()


if __name__ == '__main__':
    # python3 train.py --train chm13-data/chr19/ --valid chm13-data/chr19/ --overfit --asm hifiasm > hq-training
    # python3 train.py --train chm13htert-data/chr19/ --valid chm13htert-data/chr19/ --overfit --asm hifiasm > test-training
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='Path to the dataset')
    parser.add_argument('--valid', type=str, help='Path to the dataset')
    parser.add_argument('--asm', type=str, help='Assembler used')
    parser.add_argument('--name', type=str, default=None, help='Name for the model')
    parser.add_argument('--overfit', action='store_true', help='Overfit on the training data')
    parser.add_argument('--resume', action='store_true', help='Resume in case training failed')
    parser.add_argument('--finetune', action='store_true', help='Finetune a trained model')
    parser.add_argument('--ft_model', type=str, help='Path to the model for fine-tuning')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate for the model')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    # parser.add_argument('--savedir', type=str, default=None, help='Directory to save the model and the checkpoints')
    args = parser.parse_args()

    train(train_path=args.train, valid_path=args.valid, assembler=args.asm, out=args.name, overfit=args.overfit, \
          dropout=args.dropout, seed=args.seed, resume=args.resume, finetune=args.finetune, ft_model=args.ft_model)
