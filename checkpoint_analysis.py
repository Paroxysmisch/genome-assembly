import torch

from arabidopsis import ArabidopsisDataset
from model import Model

# model = Model.load_from_checkpoint(
#     "lightning_logs/version_52/checkpoints/epoch=19-step=4000.ckpt",
# )
model = Model.load_from_checkpoint(
    "lightning_logs/version_79/checkpoints/epoch=0-step=100.ckpt",
)
model.cpu()
model.eval()

arabidopsis_dataset = ArabidopsisDataset(root="./arabidopsis-dataset")
full_graph = arabidopsis_dataset[4]
print(f"Optimal pos_weight: {arabidopsis_dataset.get_optimal_pos_weight(20)}")
# test_loader = arabidopsis_dataset.get_clustered_data_loader(chromosome=20, num_parts=200)
test_loader = arabidopsis_dataset.get_graphsaint_data_loader(chromosome=20, batch_size=200, walk_length=5)
subgraph, reads = next(iter(test_loader))
subgraph, reads = next(iter(test_loader))
print(subgraph.edge_index[:, 0])
print(subgraph.random_feature[0, :])
original_node_idx = subgraph.read_index[subgraph.edge_index[:, 0]]
print(subgraph.read_index[subgraph.edge_index[:, 0]])
test = (full_graph.edge_index == original_node_idx[:, None])
print(full_graph.random_feature[test.all(0)])
torch.set_printoptions(threshold=10_000)

prediction = model((subgraph, reads))
prediction = torch.nn.functional.sigmoid(prediction)
# post_processed = torch.where(prediction > 0.90, 1, 0).flatten()
post_processed = prediction.flatten()
# torch.set_printoptions(threshold=10_000)

# reads_dataset = arabidopsis_dataset.reads_dataset_factory(3)
# lengths = [len(reads_dataset[i]) for i in range(len(reads_dataset))]
# histogram = torch.histogram(torch.tensor(lengths, dtype=torch.float32), 20)
# print(histogram)

print(post_processed)
# print(torch.equal(post_processed, subgraph.target))
print((post_processed < 0.5).nonzero().flatten())
print((subgraph.target == 0).nonzero().flatten())
print(prediction[(subgraph.target == 0).nonzero().flatten()])
# print(len(subgraph.target))
