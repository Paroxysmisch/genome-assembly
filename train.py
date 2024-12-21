import lightning as L
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader

from dataset import *
from lightning_modules import Model, TrainingConfig

seed_everything(42, workers=True)

train_loader = DataLoader(
    load_partitioned_dataset(Dataset.CHM13, 19),
    batch_size=1,
    collate_fn=lambda single_graph_in_list: single_graph_in_list[0],
    num_workers=16,
)
model = Model(TrainingConfig())

trainer = L.Trainer(max_epochs=100, log_every_n_steps=1, deterministic=True)
trainer.fit(model=model, train_dataloaders=train_loader)
