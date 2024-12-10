import torch.nn as nn

from arabidopsis import ArabidopsisDataset
from decoder import Decoder
from encoder import Encoder
from processor import GCN_Processor


class Model(nn.Module):
    def __init__(self, hidden_dim, num_processor_layers):
        super().__init__()
        self.enc = Encoder(hidden_dim)
        self.dec = Decoder(hidden_dim)
        self.proc = GCN_Processor(num_processor_layers, hidden_dim)

    def forward(self, data, read_data_batched):
        encoded = self.enc(data, read_data_batched)
        processed = self.proc(*encoded, data)
        decoded = self.dec(*processed, data)

        return decoded


test = ArabidopsisDataset(root="./arabidopsis-dataset")
data_iter = iter(test.get_clustered_data_loader(3, 256))
first_subgraph, first_read_data_batched = next(data_iter)
model = Model(64, 6)

print(model(first_subgraph, first_read_data_batched))
