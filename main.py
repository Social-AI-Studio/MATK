import os
import json
import logging
import hydra
import importlib
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
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model's Parameters: {total_parameters}")
    logging.info(f"Model's Trainable Parameters: {trainable_parameters}")

    ## Sanity Checks
    datamodule = datamodule_class(dataset_cfg=cfg.dataset, **cfg.datamodule)
    datamodule.setup(stage="fit")
    logging.info("Logging an example record of the dataset")
    logging.info(datamodule.train_dataloader().dataset[0])

    logging.info(next(iter(datamodule.train_dataloader())))

    trainer = Trainer(**cfg.trainer)

    if cfg.action == "fit":
        logging.info("Training model...")
        trainer.fit(model, datamodule)

        logging.info("Evaluating model - validate...")
        trainer.validate(model, datamodule)

        logging.info("Evaluating model - test...")
        trainer.test(model, datamodule, ckpt_path='best')
    elif cfg.action == "test":
        logging.info("Evaluating model...")
        model = model_class.load_from_checkpoint(
            checkpoint_path=cfg.model_checkpoint,
        )
        trainer.test(model, datamodule)
    elif cfg.action == "predict":
        logging.info("Performing model inference...")
        model = model_class.load_from_checkpoint(
            checkpoint_path=cfg.model_checkpoint,
        )
        # trainer.test(model, datamodule)
        predictions = trainer.predict(model, datamodule)


        img_filenames = []
        hate_preds = []
        hate_labels = []
        for d in predictions:
            hate_preds += d['hate_preds']
            hate_labels += d['hate_labels']
            img_filenames += d['img']
        
        # print(os.getcwd())
        result_filepath = os.path.join(os.getcwd(), f"{cfg.experiment_name}.json")
        with open(result_filepath, "w+") as f:
            json.dump({
                "img": img_filenames,
                "hate_preds": hate_preds,
                "hate_labels": hate_labels
            }, f)
    else:
        raise Exception(f"Requested action {cfg.action} unimplemented")


if __name__ == "__main__":
    main()