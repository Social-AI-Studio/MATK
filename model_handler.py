import yaml

class ModelHandler:
    def __init__(self, model, dataset, task=None):
        self.model = model
        self.dataset = dataset
        self.task = task

        with open('configs/model2dataset_mapping.yaml', 'r') as file:
            self.DATASET_CONFIGS = yaml.safe_load(file)

    def get_cls_dict(self):
        if self.dataset in self.DATASET_CONFIGS:
            if self.task:
                dataset_config = self.DATASET_CONFIGS[self.dataset][self.task]
            else:
                dataset_config = self.DATASET_CONFIGS[self.dataset]
                
            if self.model in dataset_config:
                return dataset_config[self.model]
            else:
                raise Exception(
                    f"No model {self.model} implemented for dataset {self.dataset}")
        else:
            raise Exception(f"Dataset {self.dataset} not implemented")
