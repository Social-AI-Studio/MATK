import importlib

def import_class(dataset_class):
    package_name = ".".join(dataset_class.split(".")[:-1])
    class_name = dataset_class.split(".")[-1]
    m = importlib.import_module(package_name)
    
    return getattr(m, class_name)