import yaml
class DatasetHandler:
    def __init__(self, datasets):
        with open(datasets, 'r') as file:
            self.datasets = yaml.safe_load(file)
        
    def get_dataset_info(self, dataset_name):
        if dataset_name in self.datasets["datasets"]:
            return self.datasets["datasets"][dataset_name]
        else:
            return None
