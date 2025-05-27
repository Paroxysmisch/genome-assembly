from simplified_new_train import TrainingConfig, SubgraphDataset
from inference import SubgraphDatasetNoMetis
import os
import torch
import utils
import statistics
from models import ModelType


cfg = TrainingConfig()
cfg.device = "cpu"
cfg.num_nodes_per_cluster = 2000

# Configuration for dataset loading
# cfg.validation_chromosomes = [9]
cfg.validation_chromosomes = [21]
# cfg.num_nodes_per_cluster = 2000
cfg.data_dir = "chm13htert-data/"
ds_valid = SubgraphDatasetNoMetis(cfg, False, 100, 100, 2000, shuffle=False)
# ds_valid = SubgraphDataset(cfg, False, 100, 100, cfg.num_nodes_per_cluster, shuffle=False)

# Configuration for model loading
cfg.model_type = ModelType.SymGatedGCNRandomEdge
# cfg.training_chromosomes = [19]
# cfg.validation_chromosomes = [11]
cfg.training_chromosomes = [15]
cfg.validation_chromosomes = [22]
cfg.num_nodes_per_cluster = 600
cfg.use_cuda = False
# cfg.granola = True
wandb_project = "mamba"


validation_pos_to_neg_ratio = ds_valid.pos_to_neg_ratio

model = cfg.model_type.value(cfg.num_node_features, cfg.num_edge_features, cfg.num_intermediate_hidden_features, cfg.num_hidden_features, cfg.num_layers, cfg.num_hidden_edge_scores, cfg.batch_norm, dropout=None, rnf=cfg.rnf, granola=cfg.granola, use_cuda=cfg.use_cuda)
model.to(cfg.device)

accs, precisions, recalls, f1s = [], [], [], []
accs_inv, precisions_inv, recalls_inv, f1s_inv = [], [], [], []


for seed in range(5):
    models_path = os.path.abspath(f"artifacts/models/{wandb_project}")
    out = f'model={cfg.model_type.value.__name__}_seed={seed}_train={cfg.training_chromosomes[0]}_valid={cfg.validation_chromosomes[0]}_data={cfg.data_dir[:-1]}_nodes={cfg.num_nodes_per_cluster}'
    model_path = os.path.join(models_path, f'model_{out}.pt')
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    device = cfg.device

    for batch in ds_valid:
        sub_g, pe, e, rev_sub_g, rev_pe, rev_e, labels = batch
        sub_g, pe, e, rev_sub_g, rev_pe, rev_e, labels = sub_g.to(device), pe.to(device), e.to(device), rev_sub_g.to(device), rev_pe.to(device), rev_e.to(device), labels.to(device)

        with torch.no_grad():
            org_scores = model(sub_g, pe, e).squeeze(-1)

        edge_predictions = org_scores
        edge_labels = labels

        TP, TN, FP, FN = utils.calculate_tfpn(edge_predictions, edge_labels)
        acc, precision, recall, f1 =  utils.calculate_metrics(TP, TN, FP, FN)
        acc_inv, precision_inv, recall_inv, f1_inv =  utils.calculate_metrics_inverse(TP, TN, FP, FN)

        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accs_inv.append(acc_inv)
        precisions_inv.append(precision_inv)
        recalls_inv.append(recall_inv)
        f1s_inv.append(f1_inv)

print(f"acc: {statistics.mean(accs), statistics.stdev(accs)}, precision: {statistics.mean(precisions), statistics.stdev(precisions)}, recall: {statistics.mean(recalls), statistics.stdev(recalls)}, f1: {statistics.mean(f1s), statistics.stdev(f1s)}")
print(f"acc_inv: {statistics.mean(accs_inv), statistics.stdev(accs_inv)}, precision_inv: {statistics.mean(precisions_inv), statistics.stdev(precisions_inv)}, recall_inv: {statistics.mean(recalls_inv), statistics.stdev(recalls_inv)}, f1_inv: {statistics.mean(f1s_inv), statistics.stdev(f1s_inv)}")
