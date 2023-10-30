import importlib
import torch

def import_class(dataset_class):
    package_name = ".".join(dataset_class.split(".")[:-1])
    class_name = dataset_class.split(".")[-1]
    m = importlib.import_module(package_name)
    
    return getattr(m, class_name)

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
