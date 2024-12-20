import pickle
import tarfile

import torch
from dgl import load_graphs
from torch.utils.data.dataset import Dataset
from torch_geometric.data import (Data, GraphSAINTEdgeSampler, InMemoryDataset,
                                  download_google_url)
from torch_geometric.loader.cluster import ClusterData, ClusterLoader
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import from_dgl


class ArabidopsisDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["arabidopsis-data.tar"]

    @property
    def processed_file_names(self):
        return ["arabidopsis-data.pt"]

    def download(self):
        download_google_url(
            id="1QqfLJcs0LYWx2tr1rYAwKO754OVEcHHM",
            folder=self.raw_dir,
            filename="arabidopsis-data.tar",
        )

    def process(self):
        archive = tarfile.open(self.raw_dir + "/arabidopsis-data.tar")
        archive.extractall(self.raw_dir)
        archive.close()

        # Read data into huge `Data` list.
        data_list = []

        chromosomes = [3, 4, 5, 19, 20]
        dgls = [
            "chr" + str(chromosome_number) + ".dgl" for chromosome_number in chromosomes
        ]

        for dgl in dgls:
            (graph,), _ = load_graphs(self.raw_dir + "/" + dgl)
            data = from_dgl(graph)

            # Explicitly set num_nodes and num_edges
            # as PyG assume y has a node target, but here it is an edge target
            data.num_nodes = len(data.read_length)
            # Rename y to target to avoid further problems
            data.target = data.y
            del data.y

            # Tensor of node identifier to fetch the read data
            data.read_index = torch.arange(data.num_nodes)
            data.edge_attr = data.overlap_similarity.unsqueeze(1)
            # data.edge_attr = data.target.unsqueeze(-1)
            data.edge_transposed = torch.transpose(data.edge_index, 1, 0)
            data.random_feature = data.edge_transposed
            # data.random_feature = torch.randn((data.edge_index.shape()[0] ,4))
            del data.overlap_similarity

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])

    def reads_dataset_factory(self, chromosome, transform=None):
        class ArabidopsisReadsDataset(Dataset):
            def __init__(self, root, transform):
                self.root = root
                self.transform = transform

                self.reads_file = (
                    self.root + "/raw/chr" + str(chromosome) + "_reads.pkl"
                )

                with open(self.reads_file, "rb") as f:
                    self.reads_dict = pickle.load(f)

            def __len__(self):
                return len(self.reads_dict)

            def __getitem__(self, idx):
                sample = self.reads_dict[idx]

                # Convert each sample read into one-hot encoded matrix
                indices = []
                for char in sample:
                    match char:
                        case "A":
                            indices.append(0)
                        case "T":
                            indices.append(1)
                        case "C":
                            indices.append(2)
                        case "G":
                            indices.append(3)
                sample = torch.nn.functional.one_hot(
                    torch.tensor(indices), num_classes=4
                ).float()

                # Transform operates on one-hot encoded tensor
                if self.transform:
                    sample = self.transform(sample)

                return sample

            def gen_batch(self, read_indices):
                # Allows passing both a list and a PyTorch tensor to read_indices
                if torch.is_tensor(read_indices):
                    read_indices = read_indices.tolist()

                return ArabidopsisDataset.collate_reads_fn(
                    [self.__getitem__(idx) for idx in read_indices]
                )

        return ArabidopsisReadsDataset(self.root, transform)

    @staticmethod
    def collate_reads_fn(batch):
        max_length = max(item.size(0) for item in batch)

        # Pad each sequence to max_length
        padded_batch = torch.stack(
            [
                torch.nn.functional.pad(
                    item,
                    (
                        0,
                        0,
                        0,
                        max_length - item.size(0),
                    ),  # Pad length dimension, concatenating 0s at the bottom
                )
                for item in batch
            ]
        )

        return padded_batch

    def get_graphsaint_data_loader(self, chromosome, batch_size, walk_length=2):
        chromosomes = [3, 4, 5, 19, 20]
        full_chromosome_data = self[chromosomes.index(chromosome)]
        reads_dataset = self.reads_dataset_factory(chromosome)

        return map(
            lambda subgraph: (subgraph, reads_dataset.gen_batch(subgraph.read_index)),
            GraphSAINTEdgeSampler(full_chromosome_data, batch_size, walk_length),
        )

    def get_clustered_data_loader(self, chromosome, num_parts):
        chromosomes = [3, 4, 5, 19, 20]
        full_chromosome_data = self[chromosomes.index(chromosome)]

        # current_index = full_chromosome_data.edge_index
        # orig_num_edges = full_chromosome_data.num_edges
        # new_data = Data(
        #     read_index=full_chromosome_data.read_index,
        #     edge_index=torch.concatenate(
        #         [current_index, torch.stack([current_index[1], current_index[0]])]
        #     ),
        #     edge_attr=torch.concatenate(
        #         [full_chromosome_data.edge_attr, full_chromosome_data.edge_attr]
        #     ),
        #     original_edge = torch.concatenate(
        #         [torch.ones(orig_num_edges), torch.zeros(orig_num_edges)]
        #     ),
        #     num_nodes=full_chromosome_data.num_nodes,
        # )

        full_chromosome_data = ToUndirected()(full_chromosome_data)
        reads_dataset = self.reads_dataset_factory(chromosome)

        class ClusterLoaderWrapper:
            def __init__(self, full_chromosome_data, num_parts):
                self.cluster_data = ClusterData(full_chromosome_data, num_parts)
                self.cluster_loader = ClusterLoader(self.cluster_data)
                self.reads_dataset = reads_dataset

            def __len__(self):
                return len(self.cluster_data)

            def __iter__(self):
                self.cluster_loader_iter = iter(self.cluster_loader)
                return self

            def __next__(self):
                subgraph = next(self.cluster_loader_iter)
                return (subgraph, self.reads_dataset.gen_batch(subgraph.read_index))

        return ClusterLoaderWrapper(full_chromosome_data, num_parts)

    def get_optimal_pos_weight(self, chromosome):
        chromosomes = [3, 4, 5, 19, 20]
        full_chromosome_data = self[chromosomes.index(chromosome)]
        num_positive_edges = torch.count_nonzero(full_chromosome_data.target)
        num_negative_edges = len(full_chromosome_data.target) - num_positive_edges
        print(num_positive_edges)
        print(num_negative_edges)

        return num_negative_edges / num_positive_edges


# test = ArabidopsisDataset(root="./arabidopsis-dataset")
# print(test[0])
# loader = test.get_graphsaint_data_loader(20, 200, 5)
# for subgraph in loader:
#     print(subgraph)

# print(test[0])
# reads_dataset = test.reads_dataset_factory(3)
# print(reads_dataset.__getitem__(0)[:10])
#
# test2 = reads_dataset.gen_batch(torch.tensor([0, 1, 2, 3]))
# print(test2)
# print([len(reads_dataset.__getitem__(read)) for read in [0, 1, 2, 3]])
