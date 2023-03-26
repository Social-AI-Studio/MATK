from .fhm import FHMDataModule

def load_datamodule(dataset_name, model_class_or_path, **kwargs):
    if "fhm" in dataset_name:
        return FHMDataModule(dataset_name, model_class_or_path, **kwargs)
    else:
        raise NotImplementedError(f"'{dataset_name}' datamodule not implemented")