import lightning as L

from dataset import *
from lightning_modules import Model

train_loader = iter(load_partitioned_dataset(Dataset.CHM13, 19))
model = Model(pos_weight=0.008)

trainer = L.Trainer(max_epochs=1, log_every_n_steps=1)
trainer.fit(model=model, train_dataloaders=train_loader)
