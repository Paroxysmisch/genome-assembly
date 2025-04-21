import lightning as L
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader
from typing import Type

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
    wandb_logger = WandbLogger(project="genome-assembly", name=cfg.model_type.value.__name__)

    seed_everything(42, workers=True)

    train_loader = DataLoader(
        load_partitioned_dataset(Dataset.CHM13, 19),
        batch_size=1,
        collate_fn=lambda single_graph_in_list: single_graph_in_list[0],
        num_workers=16,
        shuffle=True,
    )
    validation_loader = DataLoader(
        load_partitioned_dataset(Dataset.CHM13, 18),
        batch_size=1,
        collate_fn=lambda single_graph_in_list: single_graph_in_list[0],
        num_workers=16,
        shuffle=False,
    )
    model = Model(cfg)

    trainer = L.Trainer(max_epochs=20, log_every_n_steps=1, deterministic=True, logger=wandb_logger, check_val_every_n_epoch=5)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader)


if __name__ == "__main__":
    main()

