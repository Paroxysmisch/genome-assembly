import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import wandb
from pathlib import Path
from tqdm import tqdm
import itertools
import os

api = wandb.Api()

project_name = "base"

model_types = ["SymGatedGCNModel", "GATModel", "SymGATModel"]
train = "19"
valid = "11"
data = "chm13htert-data"
nodes = "2000"

model_type_mapping = {
    "SymGatedGCNModel": "SymGatedGCN",
    "GATModel": "GAT+Edge",
    "SymGATModel": "SymGAT+Edge",
    "SymGatedGCNMambaModel": "SymGatedGCN+Mamba",
    "SymGatedGCNMambaOnlyModel": "SymGatedGCN+MambaEdge",
    "SymGatedGCNRandomEdgeModel": "SymGatedGCN+RandomEdge",
}

col_name_mapping = {
    "epoch": "Epoch",
    "variable": "Metric",
    "value": "Rate",
    "validation_loss_epoch": "Validation Loss",
    "validation_acc_epoch": "Validation Accuracy",
    "validation_recall_epoch": "Validation Recall",
    "validation_recall_inv_epoch": "Validation Recall Inverse",
    "validation_precision_epoch": "Validation Precision",
    "validation_precision_inv_epoch": "Validation Precision Inverse",
    "validation_f1_epoch": "Validation F1",
    "validation_f1_inv_epoch": "Validation F1 Inverse",
    "validation_fp_rate_epoch": "Validation FP",
    "validation_fn_rate_epoch": "Validation FN",
}

keys = ["validation_loss_epoch", "validation_acc_epoch", "validation_recall_epoch", "validation_recall_inv_epoch", "validation_precision_epoch", "validation_precision_inv_epoch", "validation_f1_epoch", "validation_f1_inv_epoch", "validation_fp_rate_epoch", "validation_fn_rate_epoch"]

for key in keys:
    print(key)
    plt.figure()
    plot_name = f'key={key}_train={train}_valid={valid}_data={data}_nodes={nodes}'
    if not os.path.isdir(f'./plots/{project_name}'):
        os.makedirs(f'./plots/{project_name}')

    for model_type in model_types:
        all_data = []
        for seed in range(5):
            if seed == 2:
                continue
            filters = {
                "displayName": f'model={model_type}_seed={seed}_train={train}_valid={valid}_data={data}_nodes={nodes}'
            }
            print(filters["displayName"])

            runs = api.runs(path=f"paroxysmisch-university-of-cambridge/{project_name}", filters=filters)
            for run in runs:
                history = run.history(keys=[key], x_axis='epoch', pandas=(True), samples=200)
                all_data.append(history)
                break

        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.rename(columns=col_name_mapping)
        sns.set_style(style="whitegrid")

        ax = sns.lineplot(data=combined_df, x="Epoch", y=col_name_mapping[key], errorbar=('ci', 95), estimator='mean', err_style='band', label=model_type_mapping[model_type])

    plt.savefig(f'plots/{project_name}/{plot_name}.png', dpi=400)

