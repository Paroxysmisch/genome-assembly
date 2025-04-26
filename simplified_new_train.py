import argparse
from datetime import datetime
import os
import random

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import dgl
import wandb
from torch.utils import data

import models
import utils


def get_config():
    return {
        'checkpoints_path': 'checkpoints',
        'models_path': 'checkpoints',
        
        'tool_dir': 'vendor',
        'raven_dir': 'vendor/raven-1.8.1',
        'hifiasm_dir': 'vendor/hifiasm-0.18.8',
        'pbsim3_dir': 'vendor/pbsim3',
        
        'sample_profile_ID': '20kb-m64011_190830_220126',
        'sample_file': '',
        # 'sample_profile_ID': 'ont',
        # 'sample_file': '/home/yash/Projects/GNNome/vendor/pbsim3/CHM13_T2T_ONT_fastq_guppy_6.3.7_hac_ae.fastq',
        'sequencing_depth': 60,
    }

def get_hyperparameters():
    return {
        
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        # 'device': 'cpu',
        'seed': 1,
        'wandb_mode': 'disabled',  # switch between 'online' and 'disabled'
        'wandb_project': 'GNNome',

        'chr_overfit': 0,
        'plot_nga50_during_training': False,
        'eval_frequency': 20, 

        # Data
        'use_similarities': True,

        # Model
        'dim_latent': 64,
        'num_gnn_layers': 8,
        'node_features': 2,
        'edge_features': 2,  # Put 2 if you use similarities, 1 otherwise
        'hidden_edge_features': 16,
        'hidden_edge_scores': 64,
        'nb_pos_enc': 0,
        'type_pos_enc': 'none',
        'batch_norm': True,
        # 'dropout': 0.08,

        # Training
        'num_epochs': 250,
        'lr': 1e-4,
        'use_symmetry_loss': True,
        'alpha': 0.1,
        'num_parts_metis_train': 200,
        'num_parts_metis_eval': 200,
        'num_nodes_per_cluster': 2000,  # 2000 = max 10GB GPU memory for d=128, L=8
        'npc_lower_bound': 1,  # 0.8
        'npc_upper_bound': 1,  # 1.2
        'k_extra_hops': 1,
        'patience': 2,
        'decay': 0.95,
        'masking': True,
        'mask_frac_low': 80,   # ~ 25x
        'mask_frac_high': 100, # ~ 60x

        # Decoding
        'strategy': 'greedy',
        'num_decoding_paths': 100,
        'decode_with_labels': False,
        'load_checkpoint': True,
        'num_threads': 32,
        'B': 1,
        'len_threshold': 10,
    }


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


def train(train_path, valid_path, out, assembler, overfit=False, dropout=None, seed=None, resume=False, finetune=False, ft_model=None):
    hyperparameters = get_hyperparameters()
    if seed is None:
        seed = hyperparameters['seed']
    num_epochs = hyperparameters['num_epochs']
    num_gnn_layers = hyperparameters['num_gnn_layers']
    hidden_features = hyperparameters['dim_latent']
    nb_pos_enc = hyperparameters['nb_pos_enc']
    patience = hyperparameters['patience']
    lr = hyperparameters['lr']
    device = hyperparameters['device']
    batch_norm = hyperparameters['batch_norm']
    node_features = hyperparameters['node_features']
    edge_features = hyperparameters['edge_features']
    hidden_edge_features = hyperparameters['hidden_edge_features']
    hidden_edge_scores = hyperparameters['hidden_edge_scores']
    decay = hyperparameters['decay']
    wandb_mode = hyperparameters['wandb_mode']
    wandb_project = hyperparameters['wandb_project']
    num_nodes_per_cluster = hyperparameters['num_nodes_per_cluster']
    npc_lower_bound = hyperparameters['npc_lower_bound']
    npc_upper_bound = hyperparameters['npc_upper_bound']
    k_extra_hops = hyperparameters['k_extra_hops']
    masking = hyperparameters['masking']
    mask_frac_low = hyperparameters['mask_frac_low']
    mask_frac_high = hyperparameters['mask_frac_high']
    use_symmetry_loss = hyperparameters['use_symmetry_loss']
    alpha = hyperparameters['alpha']    

    config = get_config()
    checkpoints_path = os.path.abspath(config['checkpoints_path'])
    models_path = os.path.abspath(config['models_path'])

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
    assert train_path is not None, "train_path not specified!"
    assert valid_path is not None, "valid_path not specified!"

    # if not overfit:
    #     ds_train = AssemblyGraphDataset(train_path, assembler=assembler)
    #     ds_valid = AssemblyGraphDataset(valid_path, assembler=assembler)
    # else:
    #     ds_train = ds_valid = AssemblyGraphDataset(train_path, assembler=assembler)

    # (loaded_graph,), _ = dgl.load_graphs(os.path.join(train_path, 'hifiasm/processed/0.dgl'))
    # loaded_graph = preprocess_graph(loaded_graph)
    # loaded_graph = add_positional_encoding(loaded_graph)
    # ds_train = ds_valid = [(0, loaded_graph)]
    ds_train = SubgraphDataset()

    # pos_to_neg_ratio = sum([((torch.round(g.edata['y'])==1).sum() / (torch.round(g.edata['y'])==0).sum()).item() for idx, g in ds_train]) / len(ds_train)
    pos_to_neg_ratio = ds_train.pos_to_neg_ratio

    # model = my_models.SymGatedGCNModel(node_features, edge_features, hidden_features, hidden_edge_features, num_gnn_layers, hidden_edge_scores, batch_norm, nb_pos_enc, dropout=dropout)
    model = models.SymGatedGCNModel(node_features, edge_features, hidden_edge_features, hidden_features, num_gnn_layers, hidden_edge_scores, batch_norm, dropout=dropout)
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
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=decay, patience=patience, verbose=True)
    start_epoch = 0

    loss_per_epoch_train, loss_per_epoch_valid = [], []
    f1_inv_per_epoch_valid = []

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    elapsed = utils.timedelta_to_str(datetime.now() - time_start)
    print(f'Loading data done. Elapsed time: {elapsed}')

    try:
        with wandb.init(project=wandb_project, config=hyperparameters, mode=wandb_mode, name=out):
            wandb.watch(model, criterion, log='all', log_freq=1000)

            for epoch in range(start_epoch, num_epochs):

                train_loss_all_graphs, train_fp_rate_all_graphs, train_fn_rate_all_graphs = [], [], []
                train_acc_all_graphs, train_precision_all_graphs, train_recall_all_graphs, train_f1_all_graphs = [], [], [], []
                
                train_loss_epoch, train_fp_rate_epoch, train_fn_rate_epoch = [], [], []
                train_acc_epoch, train_precision_epoch, train_recall_epoch, train_f1_epoch = [], [], [], []
                train_acc_inv_epoch, train_precision_inv_epoch, train_recall_inv_epoch, train_f1_inv_epoch = [], [], [], []
                train_aps_epoch, train_aps_inv_epoch = [], []

                print('\n===> TRAINING\n')
                # random.shuffle(ds_train.graph_list)
                # for data in ds_train:
                for data in range(1):
                    model.train()
                    # idx, g = data
                    # breakpoint()
                    
                    # print(f'\n(TRAIN: Epoch = {epoch:3}) NEW GRAPH: index = {idx}')
                    print(f'\n(TRAIN: Epoch = {epoch:3}) NEW GRAPH: index = {-1}')

                    # if masking:
                    #     fraction = random.randint(mask_frac_low, mask_frac_high) / 100  # Fraction of nodes to be left in the graph (.85 -> ~30x, 1.0 -> 60x)
                    #     g = mask_graph_strandwise(g, fraction, device)

                    # # Number of clusters dependant on graph size!
                    # num_nodes_per_cluster_min = int(num_nodes_per_cluster * npc_lower_bound)
                    # num_nodes_per_cluster_max = int(num_nodes_per_cluster * npc_upper_bound) + 1
                    # num_nodes_for_g = torch.LongTensor(1).random_(num_nodes_per_cluster_min, num_nodes_per_cluster_max).item()
                    # num_clusters = g.num_nodes() // num_nodes_for_g + 1


                    print(f'\nUse METIS: True')
                    # print(f'Number of clusters:', num_clusters)
                    # g = g.long()
                    # d = dgl.metis_partition(g, num_clusters, extra_cached_hops=k_extra_hops)
                    # sub_gs = list(d.values())
                    # random.shuffle(sub_gs)
                    
                    # Loop over all mini-batch in the graph
                    running_loss, running_fp_rate, running_fn_rate = [], [], []
                    running_acc, running_precision, running_recall, running_f1 = [], [], [], []

                    # for sub_g in sub_gs:
                    for batch in ds_train:
                        sub_g, pe, e, rev_sub_g, rev_pe, rev_e, labels = batch
                        sub_g, pe, e, rev_sub_g, rev_pe, rev_e, labels = sub_g.to('cuda'), pe.to('cuda'), e.to('cuda'), rev_sub_g.to('cuda'), rev_pe.to('cuda'), rev_e.to('cuda'), labels.to('cuda')
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
                        # These are used for epoch mean = mean over partitions over graphs - mostly DEPRECATED
                        running_loss.append(loss.item())
                        running_fp_rate.append(fp_rate)
                        running_fn_rate.append(fn_rate)
                        running_acc.append(acc)
                        running_precision.append(precision)
                        running_recall.append(recall)
                        running_f1.append(f1)
                        
                        # These are used for epoch mean = mean over all the partitions in all the graphs
                        train_loss_epoch.append(loss.item())
                        train_fp_rate_epoch.append(fp_rate)
                        train_fn_rate_epoch.append(fn_rate)
                        train_acc_epoch.append(acc)
                        train_precision_epoch.append(precision)
                        train_recall_epoch.append(recall)
                        train_f1_epoch.append(f1)
                        
                        # Inverse metrics because F1 and them are not good for dataset with mostly positive labels
                        train_acc_inv_epoch.append(acc_inv)
                        train_precision_inv_epoch.append(precision_inv)
                        train_recall_inv_epoch.append(recall_inv)
                        train_f1_inv_epoch.append(f1_inv)

                    # Average over all mini-batches (partitions) in a single graph - mostly DEPRECATED
                    train_loss = np.mean(running_loss)
                    train_fp_rate = np.mean(running_fp_rate)
                    train_fn_rate = np.mean(running_fn_rate)
                    train_acc = np.mean(running_acc)
                    train_precision = np.mean(running_precision)
                    train_recall = np.mean(running_recall)
                    train_f1 = np.mean(running_f1)

                    # Record after each graph in the dataset - mostly DEPRECATED
                    train_loss_all_graphs.append(train_loss)
                    train_fp_rate_all_graphs.append(train_fp_rate)
                    train_fn_rate_all_graphs.append(train_fn_rate)
                    train_acc_all_graphs.append(train_acc)
                    train_precision_all_graphs.append(train_precision)
                    train_recall_all_graphs.append(train_recall)
                    train_f1_all_graphs.append(train_f1)

                # Average over all the training graphs in one epoch - mostly DEPRECATED
                train_loss_all_graphs = np.mean(train_loss_all_graphs)
                train_fp_rate_all_graphs = np.mean(train_fp_rate_all_graphs)
                train_fn_rate_all_graphs = np.mean(train_fn_rate_all_graphs)
                train_acc_all_graphs = np.mean(train_acc_all_graphs)
                train_precision_all_graphs = np.mean(train_precision_all_graphs)
                train_recall_all_graphs = np.mean(train_recall_all_graphs)
                train_f1_all_graphs = np.mean(train_f1_all_graphs)
                
                # Average over all the partitions in one epoch
                train_loss_epoch = np.mean(train_loss_epoch)
                train_fp_rate_epoch = np.mean(train_fp_rate_epoch)
                train_fn_rate_epoch = np.mean(train_fn_rate_epoch)
                train_acc_epoch = np.mean(train_acc_epoch)
                train_precision_epoch = np.mean(train_precision_epoch)
                train_recall_epoch = np.mean(train_recall_epoch)
                train_f1_epoch = np.mean(train_f1_epoch)
                
                train_acc_inv_epoch = np.mean(train_acc_inv_epoch)
                train_precision_inv_epoch = np.mean(train_precision_inv_epoch)
                train_recall_inv_epoch = np.mean(train_recall_inv_epoch)
                train_f1_inv_epoch = np.mean(train_f1_inv_epoch)

                loss_per_epoch_train.append(train_loss_epoch)
                lr_value = optimizer.param_groups[0]['lr']
                
                elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                print(f'\n==> TRAINING (all training graphs): Epoch = {epoch}')
                print(f'Loss: {train_loss_epoch:.4f}, fp_rate(GT=0): {train_fp_rate_epoch:.4f}, fn_rate(GT=1): {train_fn_rate_epoch:.4f}')
                print(f'Elapsed time: {elapsed}\n\n')

                if overfit:
                    if len(loss_per_epoch_valid) == 1 or len(loss_per_epoch_train) > 1 and loss_per_epoch_train[-1] < min(loss_per_epoch_train[:-1]):
                        torch.save(model.state_dict(), model_path)
                        print(f'Epoch {epoch}: Model saved!')
                    save_checkpoint(epoch, model, optimizer, loss_per_epoch_train[-1], 0.0, out, ckpt_path)
                    scheduler.step(train_loss_all_graphs)
                    wandb.log({'train_loss': train_loss_all_graphs, 'train_accuracy': train_acc_all_graphs, \
                               'train_precision': train_precision_all_graphs, 'lr_value': lr_value, \
                               'train_recall': train_recall_all_graphs, 'train_f1': train_f1_all_graphs, \
                               'train_fp-rate': train_fp_rate_all_graphs, 'train_fn-rate': train_fn_rate_all_graphs})

                    continue  # This will entirely skip the validation

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
