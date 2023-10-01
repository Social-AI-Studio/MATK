from torch.utils.data import DataLoader

from typing import Optional
from functools import partial
from .collators.processor import processor_collate_fn
from .utils import import_class, concatenate_labels, ConcatDataset
from transformers import AutoProcessor

import lightning.pytorch as pl

class ProcessorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_cfg: list,
        processor_class_or_path: str,
        batch_size: int,
        shuffle_train: bool,
        num_workers: int
    ):
        super().__init__()

        self.dataset_cfg = dataset_cfg
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers= num_workers

        for dataset in self.dataset_cfg:
            dataset_class = import_class(self.dataset_cfg[dataset].dataset_class)
            self.dataset_cfg[dataset].dataset_class = dataset_class

        processor = AutoProcessor.from_pretrained(processor_class_or_path)
        concat_labels = []
        for dataset in self.dataset_cfg:
            dataset_labels = self.dataset_cfg[dataset].labels
            encoded_labels = concatenate_labels(str(dataset), dataset_labels)
            concat_labels.extend(encoded_labels)
        
        self.collate_fn = partial(processor_collate_fn, processor=processor, labels=concat_labels)
        self.num_datasets = len(self.dataset_cfg)
        
    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:

            self.train = []
            for dataset in self.dataset_cfg:
                dataset_obj = self.dataset_cfg[dataset].dataset_class(
                    image_dir=self.dataset_cfg[dataset].image_dirs.train,
                    annotation_filepath=self.dataset_cfg[dataset].annotation_filepaths.train,
                    auxiliary_dicts=self.dataset_cfg[dataset].auxiliary_dicts.train,
                    labels=self.dataset_cfg[dataset].labels,
                    text_template=self.dataset_cfg[dataset].text_template
                )
                self.train.append(dataset_obj)
     

            self.validate = []
            for dataset in self.dataset_cfg:
                dataset_obj = self.dataset_cfg[dataset].dataset_class(
                    image_dir=self.dataset_cfg[dataset].image_dirs.validate,
                    annotation_filepath=self.dataset_cfg[dataset].annotation_filepaths.validate,
                    auxiliary_dicts=self.dataset_cfg[dataset].auxiliary_dicts.validate,
                    labels=self.dataset_cfg[dataset].labels,
                    text_template=self.dataset_cfg[dataset].text_template
                )
                self.validate.append(dataset_obj)
         

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            
            self.test = []
            for dataset in self.dataset_cfg:
                dataset_obj = self.dataset_cfg[dataset].dataset_class(
                    image_dir=self.dataset_cfg[dataset].image_dirs.test,
                    annotation_filepath=self.dataset_cfg[dataset].annotation_filepaths.test,
                    auxiliary_dicts=self.dataset_cfg[dataset].auxiliary_dicts.test,
                    labels=self.dataset_cfg[dataset].labels,
                    text_template=self.dataset_cfg[dataset].text_template
                )
                self.test.append(dataset_obj)


        if stage == "predict" or stage is None:
            self.predict = []
            for dataset in self.dataset_cfg:
                dataset_obj = self.dataset_cfg[dataset].dataset_class(
                    image_dir=self.dataset_cfg[dataset].image_dirs.predict,
                    annotation_filepath=self.dataset_cfg[dataset].annotation_filepaths.predict,
                    auxiliary_dicts=self.dataset_cfg[dataset].auxiliary_dicts.predict,
                    labels=self.dataset_cfg[dataset].labels,
                    text_template=self.dataset_cfg[dataset].text_template
                )
                self.predict.append(dataset_obj)

    def train_dataloader(self): # change the division - make it batch size per dataset
        return DataLoader(ConcatDataset(*self.train), batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=self.shuffle_train)
       
    def val_dataloader(self):
        return DataLoader(ConcatDataset(*self.validate), batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)
        
    def test_dataloader(self):
        return DataLoader(ConcatDataset(*self.test), batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)
        
    def predict_dataloader(self):
        return DataLoader(ConcatDataset(*self.predict), batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)
       