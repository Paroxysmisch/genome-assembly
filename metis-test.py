import arabidopsis
from torch_geometric.loader.cluster import ClusterData, ClusterLoader

test = arabidopsis.ArabidopsisDataset(root="./arabidopsis-dataset")
cluster_data = ClusterData(test[0], num_parts=10)
print(cluster_data)
cluster_loader = ClusterLoader(cluster_data)
first_subgraph = next(iter(cluster_loader))
print(first_subgraph)
print(first_subgraph.edge_index)
print(first_subgraph.read_idx, len(first_subgraph.read_idx))
