import lightning as L
from torch.utils.data import DataLoader

from dataset import *
from lightning_modules import Model

train_loader = DataLoader(
    load_partitioned_dataset(Dataset.CHM13, 19),
    batch_size=1,
    collate_fn=lambda single_graph_in_list: single_graph_in_list[0],
)
model = Model(pos_weight=0.008)

trainer = L.Trainer(max_epochs=100, log_every_n_steps=1)
trainer.fit(model=model, train_dataloaders=train_loader)
