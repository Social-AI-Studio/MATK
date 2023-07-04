import lightning.pytorch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import Trainer
import argparse
import yaml
import importlib

from model_handler import ModelHandler


def load_config(config_filename, key):

    with open(config_filename, 'r') as file:
        config = yaml.safe_load(file)

    if "models" in config_filename:
        return config["models"][key]

    if "trainer" in config_filename:
        return config["trainer"]

    if "data" in config_filename:
        return config["datamodules"][key]


def cli():
    parser = argparse.ArgumentParser("Fine-tuning")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--datamodule", type=str, required=True)
    parser.add_argument("--action", type=str, required=True)
    args = parser.parse_args()

    model_choice = args.model
    dataset_choice = args.dataset
    datamodule_choice = args.datamodule
    action_choice = args.action

    ## Loading the configs for model, datamodule and trainer from the correct files
    models_path = "configs/models.yaml"
    data_path = "configs/" + dataset_choice + "_data.yaml"
    trainer_path = "configs/trainer.yaml"

    req_model_config = load_config(models_path, model_choice)
    req_data_config = load_config(data_path, datamodule_choice)
    req_trainer_config = load_config(trainer_path, None)

    ## Handling the model to dataset config mappings - cls_dict, labels, label2word
    model_config_handler = ModelHandler(model_choice, dataset_choice)
    reqd_args = model_config_handler.get_cls_dict()
    req_model_config["init_args"].update(reqd_args)

    ## Importing the correct datamodule
    module = importlib.import_module("datamodules.modules")
    datamodule_class = getattr(module, datamodule_choice)

    ## Importing the correct model
    model_class_path = req_model_config["class_path"]
    package_name = ".".join(model_class_path.split(".")[:-1])
    class_name = model_class_path.split(".")[-1]
    m = importlib.import_module(package_name)
    model_class = getattr(m, class_name)

    ## Instantiation of model and datamodule
    model = model_class(**req_model_config["init_args"])
    datamodule = datamodule_class(**req_data_config)

    trainer = Trainer(**req_trainer_config)
    
    if action_choice == "fit":
        trainer.fit(model, datamodule)
    elif action_choice == "test":
        trainer.test(model, datamodule)
    else:
        raise Exception(f"Requested action {action_choice} Unavailable")
