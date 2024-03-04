from torch.utils.data import DataLoader

from typing import Optional
from functools import partial
from .collators.processor import processor_collate_fn
from .utils import import_class, ConcatDataset
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

        labels = []
        for dataset in self.dataset_cfg:
            # Import the individual dataset classes
            self.dataset_cfg[dataset].dataset_class = import_class(
                self.dataset_cfg[dataset].dataset_class
            )

            # Retrieve labels
            labels.extend(self.dataset_cfg[dataset].labels)

        # Partially load the collate functions
        processor = AutoProcessor.from_pretrained(processor_class_or_path)
        self.collate_fn = partial(processor_collate_fn, processor=processor, labels=labels)
        
    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:
            
            self.train = []
            self.validate = []
            for dataset in self.dataset_cfg:
                cfg = self.dataset_cfg[dataset]
                dataset_obj = cfg.dataset_class(
                    img_dir=cfg.image_dirs.train,
                    annotation_filepath=cfg.annotation_filepaths.train,
                    auxiliary_dicts=cfg.auxiliary_dicts.train,
                    text_template=cfg.text_template
                )
                self.train.append(dataset_obj)
     
                dataset_obj = cfg.dataset_class(
                    img_dir=cfg.image_dirs.validate,
                    annotation_filepath=cfg.annotation_filepaths.validate,
                    auxiliary_dicts=cfg.auxiliary_dicts.validate,
                    text_template=cfg.text_template
                )
                self.validate.append(dataset_obj)
         

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = []
            for dataset in self.dataset_cfg:
                cfg = self.dataset_cfg[dataset]
                dataset_obj = cfg.dataset_class(
                    img_dir=cfg.image_dirs.test,
                    annotation_filepath=cfg.annotation_filepaths.test,
                    auxiliary_dicts=cfg.auxiliary_dicts.test,
                    text_template=cfg.text_template
                )
                self.test.append(dataset_obj)


        if stage == "predict" or stage is None:
            self.predict = []
            for dataset in self.dataset_cfg:
                cfg = self.dataset_cfg[dataset]
                dataset_obj = cfg.dataset_class(
                    img_dir=cfg.image_dirs.predict,
                    annotation_filepath=cfg.annotation_filepaths.predict,
                    auxiliary_dicts=cfg.auxiliary_dicts.predict,
                    text_template=cfg.text_template
                )
                self.predict.append(dataset_obj)

    def train_dataloader(self):
        return DataLoader(ConcatDataset(*self.train), batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=self.shuffle_train)
       
    def val_dataloader(self):
        return DataLoader(ConcatDataset(*self.validate), batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)
        
    def test_dataloader(self):
        return DataLoader(ConcatDataset(*self.test), batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)
        
    def predict_dataloader(self):
        return DataLoader(ConcatDataset(*self.predict), batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)
       
