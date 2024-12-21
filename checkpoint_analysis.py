import torch
from torch.utils.data import DataLoader

from dataset import Dataset, load_partitioned_dataset
from lightning_modules import Model

model = Model.load_from_checkpoint(
    "lightning_logs/version_96/checkpoints/epoch=99-step=12800.ckpt",
)
model.cpu()
model.eval()

test_loader = DataLoader(
    load_partitioned_dataset(Dataset.CHM13, 19, [14]),
    batch_size=1,
    collate_fn=lambda single_graph_in_list: single_graph_in_list[0],
)
subgraph = next(iter(test_loader))
torch.set_printoptions(threshold=10_000)

prediction = model(subgraph)
prediction = torch.nn.functional.sigmoid(prediction)
post_processed = prediction.flatten()
print("Model output:")
print(post_processed)

print("Model predictions:")
print((post_processed < 0.1).nonzero().flatten())
print("Model prediction probabilities:")
print(post_processed[post_processed < 0.1])
print("True target:")
print((subgraph.edata["y"] == 0).nonzero().flatten())
print("Model prediction for true 0-edges:")
print(prediction[(subgraph.edata["y"] == 0).nonzero().flatten()])
