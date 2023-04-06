from .fhm import FHMDataModule, FHMT5DataModule

def load_datamodule(dataset_name, model_class_or_path, **kwargs):
    if "fhm" in dataset_name:
        return FHMT5DataModule(dataset_name, model_class_or_path, **kwargs)
    else:
        raise NotImplementedError(f"'{dataset_name}' datamodule not implemented")