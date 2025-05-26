import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import wandb
from pathlib import Path
from tqdm import tqdm
import itertools

api = wandb.Api()

project_name = "mamba"

def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size), 'valid') / window_size 

window_size = 25 # Number of epochs to calculate moving average over

model_type = "SymGatedGCNModel"
train = "19"
valid = "11"
data = "chm13htert-data"
nodes = "600"

all_data = []
col_name_mapping = {
    "epoch": "Epoch",
    "variable": "Metric",
    "value": "Rate",
    "validation_loss_epoch": "Validation Loss",
    "validation_recall_epoch": "Validation Recall",
    "validation_recall_inv_epoch": "Validation Recall Inverse",
    "validation_precision_epoch": "Validation Precision",
    "validation_precision_inv_epoch": "Validation Precision Inverse",
    "validation_fp_rate_epoch": "Validation FP",
    "validation_fn_rate_epoch": "Validation FN",
}

for seed in range(5):
    filters = {
        "displayName": f'model={model_type}_seed={seed}_train={train}_valid={valid}_data={data}_nodes={nodes}'
    }
    print(filters["displayName"])

    runs = api.runs(path=f"paroxysmisch-university-of-cambridge/{project_name}", filters=filters)
    for run in runs:
        history = run.history(keys=["validation_recall_epoch", "validation_recall_inv_epoch", "validation_precision_epoch", "validation_precision_inv_epoch", "validation_fp_rate_epoch", "validation_fn_rate_epoch"], x_axis='epoch', pandas=(True), samples=200)
        all_data.append(history)
        break

combined_df = pd.concat(all_data, ignore_index=True)
combined_df = combined_df.rename(columns=col_name_mapping)
sns.set_style(style="whitegrid")
plot_name = f'model={model_type}_train={train}_valid={valid}_data={data}_nodes={nodes}'

plt.figure()
plt.ylim(0, 1)
recall_df = combined_df[["Epoch", "Validation Recall", "Validation Recall Inverse"]]
recall_df = pd.melt(recall_df, ["Epoch"], var_name="Metric", value_name="Rate")
ax = sns.lineplot(data=recall_df, x="Epoch", y="Rate", errorbar=('ci', 95), estimator='mean', err_style='band', hue="Metric")
plt.savefig(f'{plot_name}_recall.png', dpi=400)

plt.figure()
plt.ylim(0, 1)
precision_df = combined_df[["Epoch", "Validation Precision", "Validation Precision Inverse"]]
precision_df = pd.melt(precision_df, ["Epoch"], var_name="Metric", value_name="Rate")
ax = sns.lineplot(data=precision_df, x="Epoch", y="Rate", errorbar=('ci', 95), estimator='mean', err_style='band', hue="Metric")
plt.savefig(f'{plot_name}_precision.png', dpi=400)

plt.figure()
plt.ylim(0, 1)
pos_neg_df = combined_df[["Epoch", "Validation FP", "Validation FN"]]
pos_neg_df = pd.melt(pos_neg_df, ["Epoch"], var_name="Metric", value_name="Rate")
ax = sns.lineplot(data=pos_neg_df, x="Epoch", y="Rate", errorbar=('ci', 95), estimator='mean', err_style='band', hue="Metric")
plt.savefig(f'{plot_name}_pos_neg.png', dpi=400)

