import torch

from arabidopsis import ArabidopsisDataset
from model import Model

model = Model.load_from_checkpoint(
    "lightning_logs/version_1/checkpoints/epoch=10-step=2200.ckpt",
)
model.cpu()
model.eval()

arabidopsis_dataset = ArabidopsisDataset(root="./arabidopsis-dataset")
test_loader = arabidopsis_dataset.get_clustered_data_loader(chromosome=4, num_parts=128)
subgraph, reads = next(iter(test_loader))

prediction = model((subgraph, reads))
prediction = torch.nn.functional.sigmoid(prediction)
# post_processed = torch.where(prediction > 0.90, 1, 0).flatten()
post_processed = prediction.flatten()
torch.set_printoptions(threshold=10_000)
print(post_processed)
print(torch.equal(post_processed, subgraph.target))
print(subgraph.target)
print(len(subgraph.target))
