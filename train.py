import lightning as L

from arabidopsis import ArabidopsisDataset
from model import Model

arabidopsis_dataset = ArabidopsisDataset(root="./arabidopsis-dataset")
train_loader = arabidopsis_dataset.get_clustered_data_loader(
    chromosome=3, num_parts=512
)
model = Model(hidden_dim=32, num_processor_layers=3)

trainer = L.Trainer(limit_train_batches=10, max_epochs=20, log_every_n_steps=1)
trainer.fit(model=model, train_dataloaders=train_loader)
