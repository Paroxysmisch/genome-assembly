import torch
from torch.utils.data import DataLoader

import utils
from dataset import Dataset, load_partitioned_dataset
from lightning_modules import Model, calculate_node_and_edge_features, TrainingConfig

training_config = TrainingConfig()
model = Model.load_from_checkpoint(
    # "genome-assembly/q7lpeps6/checkpoints/epoch=249-step=1250.ckpt",
    "genome-assembly/m44zkhhv/checkpoints/epoch=23-step=3072.ckpt",
    training_config=training_config
    # "lightning_logs/version_112/checkpoints/epoch=19-step=2560.ckpt",
    # "lightning_logs/version_115/checkpoints/epoch=19-step=2560.ckpt",
)
# model.cpu()
# model.eval()
torch.set_printoptions(threshold=10_000)


def test_fn(subgraph):
    print(model.device, subgraph.device)
    prediction = model(subgraph)
    print(f"Model output (unprocessed):")
    print(prediction)
    prediction = torch.nn.functional.sigmoid(prediction)
    print(f"Model output (unprocessed 2):")
    print(prediction)
    post_processed = prediction.flatten()
    print(f"Model output:")
    print(post_processed)

    print("Model predictions:")
    print((post_processed < 0.5).nonzero().flatten())
    print("Model prediction probabilities:")
    print(post_processed[post_processed < 0.5])
    print("True target:")
    print((subgraph.edata["y"] == 0).nonzero().flatten())
    print("Model prediction for true 0-edges:")
    print(prediction[(subgraph.edata["y"] == 0).nonzero().flatten()])
    if len((subgraph.edata["y"] == 0).nonzero().flatten()):
        breakpoint()


chromosome = 18
test_loader = DataLoader(
    load_partitioned_dataset(Dataset.CHM13, [chromosome], [21]),  # 21, 80
    batch_size=1,
    collate_fn=lambda single_graph_in_list: single_graph_in_list[0],
)
subgraph = next(iter(test_loader))
subgraph = subgraph.to(model.device)
test_fn(subgraph)
breakpoint()

test_loader = DataLoader(
    load_partitioned_dataset(Dataset.CHM13, [chromosome], [80]),  # 21, 80
    batch_size=1,
    collate_fn=lambda single_graph_in_list: single_graph_in_list[0],
)
subgraph = next(iter(test_loader))
subgraph = subgraph.to(model.device)
test_fn(subgraph)
breakpoint()

test_loader = DataLoader(
    load_partitioned_dataset(Dataset.CHM13, [chromosome]),
    batch_size=1,
    collate_fn=lambda single_graph_in_list: single_graph_in_list[0],
)
total_f1 = 0
for subgraph in test_loader:
    subgraph = subgraph.to(model.device)

    org_scores = model(subgraph).squeeze(-1)
    labels = subgraph.edata["y"]

    TP, TN, FP, FN = utils.calculate_tfpn(
        edge_predictions=org_scores, edge_labels=labels
    )
    accuracy, precision, recall, f1 = utils.calculate_metrics(TP, TN, FP, FN)
    total_f1 += f1
print(f"F1 score: {total_f1}")

# for i in range(32):
#     test_loader = DataLoader(
#         load_partitioned_dataset(Dataset.CHM13, chromosome, [i]),
#         batch_size=1,
#         collate_fn=lambda single_graph_in_list: single_graph_in_list[0],
#     )
#     for subgraph in test_loader:
#         print(f"Testing i={i}:")
#         subgraph = subgraph.to(model.device)
#         test_fn(subgraph)
