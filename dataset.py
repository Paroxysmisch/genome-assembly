import os
import pickle
from enum import Enum
import random

import dgl
import torch
from dgl import load_graphs
from torch.utils import data
from tqdm import tqdm


class Dataset(Enum):
    CHM13 = "chm13-dataset"
    CHM13htert = "chm13htert-dataset"


def encode_read(reads_dict, read_index):
    sample = reads_dict[read_index]

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
    sample = torch.nn.functional.one_hot(torch.tensor(indices), num_classes=4).float()

    return sample


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


def gen_batch(reads_dict, read_indices):
    # Allows passing both a list and a PyTorch tensor to read_indices
    if torch.is_tensor(read_indices):
        read_indices = read_indices.tolist()

    return collate_reads_fn(
        [encode_read(reads_dict, read_index) for read_index in read_indices]
    )


def calculate_node_and_edge_features(graph, sub_g):
    ol_len = graph.edata["overlap_length"][sub_g.edata[dgl.EID]].float()
    ol_len = (ol_len - ol_len.mean()) / ol_len.std()
    e = ol_len.unsqueeze(-1)

    pe_in = graph.in_degrees()[sub_g.ndata[dgl.NID]].float().unsqueeze(1)
    pe_in = (pe_in - pe_in.mean()) / pe_in.std()
    pe_out = graph.out_degrees()[sub_g.ndata[dgl.NID]].float().unsqueeze(1)
    pe_out = (pe_out - pe_out.mean()) / pe_out.std()

    # y = graph.edata["y"][sub_g.edata[dgl.EID]].float().unsqueeze(-1)
    pe = torch.cat((pe_in, pe_out), dim=1)

    return pe, e


def gen_partitioned_dataset(dataset: Dataset, chromosome: int, num_parts: int = 128, mask_frac_low=80, mask_frac_high=100):
    if not os.path.isdir(dataset.value):
        raise ValueError(
            f"Download {dataset.value} before generating the partitioned dataset"
        )

    raw_dir = dataset.value + "/raw/"
    partitioned_dir = dataset.value + "/partitioned/"
    graph_path = raw_dir + "chr" + str(chromosome) + ".dgl"
    reads_path = raw_dir + "chr" + str(chromosome) + "_reads.pkl"

    (graph,), _ = load_graphs(graph_path)
    reads_dict = None
    with open(reads_path, "rb") as f:
        reads_dict = pickle.load(f)


    def mask_graph_strandwise(g, fraction):
        keep_node_idx_half = torch.rand(g.num_nodes() // 2) < fraction
        keep_node_idx = torch.empty(keep_node_idx_half.size(0) * 2, dtype=keep_node_idx_half.dtype)
        keep_node_idx[0::2] = keep_node_idx_half
        keep_node_idx[1::2] = keep_node_idx_half
        sub_g = dgl.node_subgraph(g, keep_node_idx, store_ids=True)
        print(f'Masking fraction: {fraction}')
        print(f'Original graph: N={g.num_nodes()}, E={g.num_edges()}')
        print(f'Subsampled graph: N={sub_g.num_nodes()}, E={sub_g.num_edges()}')
        return sub_g

    fraction = random.randint(mask_frac_low, mask_frac_high) / 100  # Fraction of nodes to be left in the graph (.85 -> ~30x, 1.0 -> 60x)
    masked_graph = mask_graph_strandwise(graph, fraction)

    subgraphs_dict = dgl.metis_partition(masked_graph, num_parts, extra_cached_hops=1)
    assert subgraphs_dict is not None
    # Repopulate node and edge features from the original graph and add batched read data
    for _, subgraph in tqdm(subgraphs_dict.items()):
        node_ids = subgraph.ndata[dgl.NID]
        edge_ids = subgraph.edata[dgl.EID]
        for n_feature_name, n_feature in masked_graph.ndata.items():
            subgraph.ndata[n_feature_name] = n_feature[node_ids]
        for e_feature_name, e_feature in graph.edata.items():
            subgraph.edata[e_feature_name] = e_feature[edge_ids]
        # breakpoint()
        pe, e = calculate_node_and_edge_features(graph, subgraph)
        subgraph.ndata['pe'] = pe
        subgraph.edata['e'] = e
    #     subgraph.ndata["read_data"] = gen_batch(reads_dict, node_ids)

    subgraphs = list(subgraphs_dict.values())
    random.shuffle(subgraphs)
    return subgraphs

    # dgl.save_graphs(
    #     partitioned_dir + "p_chr" + str(chromosome) + ".dgl", [*subgraphs_dict.values()]
    # )


def load_partitioned_dataset(dataset: Dataset, chromosomes: list[int], partition_list=None, num_subgraphs_per_epoch=8):
    # partitioned_dir = dataset.value + "/partitioned/"
    # subgraphs = []
    # for chromosome in chromosomes:
    #     print(f"Loading partitioned dataset {dataset.value}, chromosome {chromosome}")
    #     subgraphs_for_chr, _ = load_graphs(
    #         partitioned_dir + "p_chr" + str(chromosome) + ".dgl", idx_list=partition_list
    #     )
    #     subgraphs += subgraphs_for_chr

    class SubgraphDataset(data.Dataset):
        def __init__(self):
            self.num_subgraphs_per_epoch = num_subgraphs_per_epoch
            self.subgraphs = None
            self.accessed = self.num_subgraphs_per_epoch

        def __len__(self):
            return self.num_subgraphs_per_epoch

        def __getitem__(self, idx):
            if self.accessed == self.num_subgraphs_per_epoch:
                # Regenerate subgraphs by repartitioning
                self.subgraphs = gen_partitioned_dataset(dataset, chromosomes[0], self.num_subgraphs_per_epoch)
                self.accessed = 0
            self.accessed += 1
            return self.subgraphs[idx]

    return SubgraphDataset()


    # class SubgraphDataset(data.Dataset):
    #     def __init__(self, subgraphs):
    #         self.subgraphs = subgraphs
    #
    #     def __len__(self):
    #         return len(self.subgraphs)
    #
    #     def __getitem__(self, idx):
    #         return self.subgraphs[idx]
    #
    # return SubgraphDataset(subgraphs)


# subgraphs = load_partitioned_dataset(Dataset.CHM13, 19)
# for idx in range(len(subgraphs)):
#     subgraph = subgraphs[idx]
#     if len((subgraph.edata['y'] == 0).nonzero()) > 3:
#         print(idx)
#         break
# breakpoint()

# gen_partitioned_dataset(Dataset.CHM13htert, 19, 128)
# gen_partitioned_dataset(Dataset.CHM13htert, 21, 128)
# gen_partitioned_dataset(Dataset.CHM13htert, 22, 128)
# gen_partitioned_dataset(Dataset.CHM13htert, 9, 256)
# gen_partitioned_dataset(Dataset.CHM13htert, 11, 128)
# gen_partitioned_dataset(Dataset.CHM13htert, 15, 128)
