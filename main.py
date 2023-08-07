import logging
import hydra
import importlib
import lightning.pytorch

from omegaconf import DictConfig, OmegaConf
from lightning.pytorch import Trainer, seed_everything

def get_class(class_path):
    package_name = ".".join(class_path.split(".")[:-1])
    package = importlib.import_module(package_name)
    
    class_name = class_path.split(".")[-1]
    return getattr(package, class_name)

@hydra.main(version_base=None, config_path="configs")
def main(cfg) -> None:
    cfg = hydra.utils.instantiate(cfg)

    if "seed_everything" in cfg:
        seed = cfg.seed_everything

        logging.info(f"Setting custom seed: {seed}...")
        seed_everything(seed, workers= True)

    
    model_class = get_class(class_path=cfg.model.pop("class_path"))
    datamodule_class = get_class(class_path=cfg.datamodule.pop("class_path"))

    ## Instantiation of model and datamodule
    model = model_class(metrics_cfg=cfg.metric, **cfg.model)
    datamodule = datamodule_class(dataset_cfg=cfg.dataset, **cfg.datamodule)
    trainer = Trainer(**cfg.trainer)

    ## Sanity Checks
    datamodule.setup(stage="fit")
    logging.info("Logging an example record of the dataset")
    logging.info(datamodule.train_dataloader().dataset[0])

    if cfg.action == "fit":
        logging.info("Training model...")
        trainer.fit(model, datamodule)
    elif cfg.action == "test":
        logging.info("Evaluating model...")
        model = model_class.load_from_checkpoint(
            checkpoint_path=cfg.model_checkpoint,
        )
        trainer.test(model, datamodule)
    else:
        raise Exception(f"Requested action {cfg.action} unimplemented")


if __name__ == "__main__":
    main()