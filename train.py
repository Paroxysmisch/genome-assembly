import lightning as L

from arabidopsis import ArabidopsisDataset
from model import Model

arabidopsis_dataset = ArabidopsisDataset(root="./arabidopsis-dataset")
train_loader = arabidopsis_dataset.get_clustered_data_loader(
    chromosome=3, num_parts=200
)
test_loader = arabidopsis_dataset.get_clustered_data_loader(
    chromosome=4, num_parts=200
)
model = Model(hidden_dim=64, num_processor_layers=8)

trainer = L.Trainer(limit_test_batches=0.1, max_epochs=20, log_every_n_steps=1, limit_val_batches=0.1, check_val_every_n_epoch=1)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)
