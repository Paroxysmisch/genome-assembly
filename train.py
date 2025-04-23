import lightning as L
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader
from typing import Type
import time
import wandb

from dataset import *
from lightning_modules import Model, TrainingConfig
from omegaconf import DictConfig, OmegaConf
import hydra
from pydantic import BaseModel
from pytorch_lightning.loggers import WandbLogger


def omega_to_pydantic(cfg: DictConfig, config_cls: Type[BaseModel]) -> BaseModel:
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    fields = config_cls.model_fields

    for key, field in fields.items():
        if (
            key in config_dict
            and isinstance(config_dict[key], str)
            and isinstance(field.annotation, type)
            and issubclass(field.annotation, Enum)
        ):
            config_dict[key] = field.annotation[config_dict[key]]

    return config_cls(**config_dict)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    cfg = omega_to_pydantic(cfg, TrainingConfig)
    wandb_logger = WandbLogger(project="genome-assembly", name=(cfg.model_type.value.__name__ + "_seed=" + str(cfg.seed) + "_time=" + str(time.time())), resume="never")

    seed_everything(cfg.seed, workers=True)

    model = Model(cfg)
    train_loader = DataLoader(
        load_partitioned_dataset(Dataset.CHM13htert, cfg.training_chromosomes),
        batch_size=1,
        collate_fn=lambda single_graph_in_list: single_graph_in_list[0],
        num_workers=0,
        shuffle=True,
    )
    validation_loader = DataLoader(
        load_partitioned_dataset(Dataset.CHM13htert, cfg.validation_chromosomes),
        batch_size=1,
        collate_fn=lambda single_graph_in_list: single_graph_in_list[0],
        num_workers=0,
        shuffle=False,
    )

    # trainer = L.Trainer(max_epochs=250, log_every_n_steps=1, deterministic=True, logger=wandb_logger, check_val_every_n_epoch=5)
    trainer = L.Trainer(max_epochs=250, log_every_n_steps=1, deterministic=True, logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    wandb.finish()
    print("Training Complete!")


if __name__ == "__main__":
    main()

