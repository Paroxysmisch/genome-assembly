import pickle
import tarfile

import torch
from dgl import load_graphs
from torch.utils.data.dataset import Dataset
from torch_geometric.data import InMemoryDataset, download_google_url
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

        chromosomes = [3, 4, 5]
        dgls = [
            "chr" + str(chromosome_number) + ".dgl" for chromosome_number in chromosomes
        ]

        for dgl in dgls:
            (graph,), _ = load_graphs(self.raw_dir + "/" + dgl)
            data = from_dgl(graph)
            data.num_nodes = len(
                data.read_length
            )  # Need to explicitly set as len(y) is the number of edges, which is confusing PyG
            data.read_idx = torch.arange(data.num_nodes)

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
                )

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


test = ArabidopsisDataset(root="./arabidopsis-dataset")
print(test[0])
reads_dataset = test.reads_dataset_factory(3)
print(reads_dataset.__getitem__(0)[:10])

test2 = reads_dataset.gen_batch(torch.tensor([0, 1, 2, 3]))
print(test2)
print([len(reads_dataset.__getitem__(read)) for read in [0, 1, 2, 3]])
