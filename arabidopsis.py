import tarfile
import pickle
import torch
from dgl import load_graphs
from torch_geometric.data import InMemoryDataset, download_google_url
from torch_geometric.utils import from_dgl
from torch.utils.data.dataset import Dataset


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

                self.reads_file = self.root + "/raw/chr" + str(chromosome) + "_reads.pkl"

                with open(self.reads_file, 'rb') as f:
                    self.reads_dict = pickle.load(f)

            def __len__(self):
                return len(self.reads_dict)

            def __getitem__(self, idx):
                sample = self.reads_dict[idx]

                # Convert each sample read into one-hot encoded matrix
                indices = []
                for char in sample:
                    match char:
                        case 'A':
                            indices.append(0)
                        case 'T':
                            indices.append(1)
                        case 'C':
                            indices.append(2)
                        case 'G':
                            indices.append(3)
                sample = torch.nn.functional.one_hot(torch.tensor(indices), num_classes=4)

                # Transform operates on one-hot encoded tensor
                if self.transform:
                    sample = self.transform(sample)

                return sample

        return ArabidopsisReadsDataset(self.root, transform)



test = ArabidopsisDataset(root="./arabidopsis-dataset")
print(test[0])
print(test.reads_dataset_factory(3).__getitem__(0)[:10])
