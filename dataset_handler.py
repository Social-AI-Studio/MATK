import yaml

class DatasetHandler:
    def __init__(self, dataset_file):
        try:
            with open(dataset_file, 'r') as file:
                self.datasets = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file '{dataset_file}' not found.")
        
    def get_dataset_info(self, dataset_name):
        if dataset_name in self.datasets["datasets"]:
            return self.datasets["datasets"][dataset_name]
        else:
            return None

    def get_labels(self, dataset_name, task=None):
        if dataset_name in self.datasets["datasets"]:
            if task is None: 
                return self.datasets["datasets"][dataset_name]['labels']
            return self.datasets["datasets"][dataset_name]['labels'][task]
        return None