import torch
from torch.utils.data import DataLoader

from dataset import Dataset, load_partitioned_dataset
from lightning_modules import Model

model = Model.load_from_checkpoint(
    # "lightning_logs/version_112/checkpoints/epoch=19-step=2560.ckpt",
    "lightning_logs/version_115/checkpoints/epoch=19-step=2560.ckpt",
)
# model.cpu()
# model.eval()

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

test_loader = DataLoader(
    load_partitioned_dataset(Dataset.CHM13, 19, [21]), # 21, 80
    batch_size=1,
    collate_fn=lambda single_graph_in_list: single_graph_in_list[0],
)
torch.set_printoptions(threshold=10_000)
subgraph = next(iter(test_loader))
subgraph = subgraph.to(model.device)
test_fn(subgraph)
breakpoint()

test_loader = DataLoader(
    load_partitioned_dataset(Dataset.CHM13, 19, [80]), # 21, 80
    batch_size=1,
    collate_fn=lambda single_graph_in_list: single_graph_in_list[0],
)
torch.set_printoptions(threshold=10_000)
subgraph = next(iter(test_loader))
subgraph = subgraph.to(model.device)
test_fn(subgraph)

for i in range(32):
    test_loader = DataLoader(
        load_partitioned_dataset(Dataset.CHM13, 19, [i]),
        batch_size=1,
        collate_fn=lambda single_graph_in_list: single_graph_in_list[0],
    )
    for subgraph in test_loader:
        print(f"Testing i={i}:")
        subgraph = subgraph.to(model.device)
        test_fn(subgraph)
