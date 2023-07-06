import lightning.pytorch
import yaml
import importlib

def load_config(config_filename, key):

    with open(config_filename, 'r') as file:
        config = yaml.safe_load(file)

    if "models" in config_filename:
        return config["models"][key]

    if "trainer" in config_filename:
        return config["trainers"][key]

    if "data" in config_filename:
        return config["datamodules"][key]

def load_callbacks(callback_config_list):
    callback_obj_list = []
    for callback in callback_config_list:
        package_name, class_name = callback["class_path"].rsplit(".", 1)
        m = importlib.import_module(package_name)
        callback_class = getattr(m, class_name)
        callback_object = callback_class(**callback["init_args"])
        callback_obj_list.append(callback_object)
        break
    return callback_obj_list

def load_logger(logger_config):
    
    package_name, class_name = logger_config["class_path"].rsplit(".", 1)
    m = importlib.import_module(package_name)
    logger_class = getattr(m, class_name)
    logger_object = logger_class(**logger_config["init_args"])

    return logger_object