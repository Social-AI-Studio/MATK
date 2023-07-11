import lightning.pytorch
from lightning.pytorch import Trainer
import argparse
import yaml
import importlib

from model_handler import ModelHandler
from cli_utils import load_callbacks, load_config, load_logger

def cli():
    parser = argparse.ArgumentParser("Fine-tuning")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task", type=str, required=False)
    parser.add_argument("--datamodule", type=str, required=True)
    parser.add_argument("--action", type=str, required=True)
    args = parser.parse_args()

    model_choice = args.model
    dataset_choice = args.dataset
    task = args.task
    datamodule_choice = args.datamodule
    action_choice = args.action

    ## Finding the paths for data yaml and model yaml
    models_path = "configs/models.yaml"

    if task is None:
        data_path = f"configs/data/{dataset_choice}_data.yaml"
        trainer_path = f"configs/trainers/{dataset_choice}_trainer.yaml"
    else:
        data_path = f"configs/data/{dataset_choice}_{task}_data.yaml"
        trainer_path = f"configs/trainers/{dataset_choice}_{task}_trainer.yaml"

    ## Loading the configs for model, datamodule and trainer from the correct files
    req_model_config = load_config(models_path, model_choice)
    req_data_config = load_config(data_path, datamodule_choice)
    req_trainer_config = load_config(trainer_path, model_choice)

    if task is not None:
        req_data_config.update({"task": task})

    ## Handling the model to dataset config mappings - cls_dict, labels, label2word
    model_config_handler = ModelHandler(model_choice, dataset_choice, task)
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

    ## Instantiation of callbacks
    callback_config = req_trainer_config.pop("callbacks")
    callback_config_updated = load_callbacks(callback_config)

    ## Instantiation of logger
    logger_config = req_trainer_config.pop("logger")
    logger_config_updated = load_logger(logger_config)

    req_trainer_config.update({"callbacks": callback_config_updated})
    req_trainer_config.update({"logger": logger_config_updated})
 
    trainer = Trainer(**req_trainer_config)
    
    if action_choice == "fit":
        trainer.fit(model, datamodule)
    elif action_choice == "test":
        trainer.test(model, datamodule)
    else:
        raise Exception(f"Requested action {action_choice} Unavailable")
