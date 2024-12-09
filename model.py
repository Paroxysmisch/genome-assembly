import torch
from torch_geometric.loader.cluster import ClusterData, ClusterLoader
from torch_geometric.transforms import ToUndirected

from arabidopsis import ArabidopsisDataset
from decoder import Decoder
from encoder import Encoder
from processor import GCN_Processor

hidden_dim = 32
enc = Encoder(hidden_dim)
dec = Decoder(hidden_dim)
proc = GCN_Processor(3, hidden_dim)


test = ArabidopsisDataset(root="./arabidopsis-dataset")
data = test[0]
num_edges = len(data.edge_attr)
data.edge_idx = torch.arange(num_edges)
print(data)
data = ToUndirected()(data)
print(data)
reads_dataset = test.reads_dataset_factory(3)
cluster_data = ClusterData(data, num_parts=256)
cluster_loader = ClusterLoader(cluster_data)
first_subgraph = next(iter(cluster_loader))
first_read_data_batched = reads_dataset.gen_batch(first_subgraph.read_index)
# print(first_subgraph.
print(first_read_data_batched.size())

print(first_subgraph)

encoded = enc(first_subgraph, first_read_data_batched)
processed = proc(*encoded, first_subgraph)
decoded = dec(*processed, first_subgraph)
print(decoded)
