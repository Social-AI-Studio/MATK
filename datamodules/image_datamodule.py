from torch.utils.data import DataLoader

from datamodules.collators import get_collator

from typing import Optional
from .utils import import_class

import lightning.pytorch as pl

class ImagesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_cfg: str,
        tokenizer_class_or_path: str,
        frcnn_class_or_path: str,
        batch_size: int,
        shuffle_train: bool,
        num_workers: int
    ):
        super().__init__()

        self.dataset_cfg = dataset_cfg
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers= num_workers

        self.dataset_cls = import_class(dataset_cfg.dataset_class)
        self.collate_fn = get_collator(
            tokenizer_class_or_path,
            labels=dataset_cfg.labels, 
            frcnn_class_or_path=frcnn_class_or_path
        )
        
    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:
            self.train = self.dataset_cls(
                image_dir=self.dataset_cfg.image_dirs.train,
                annotation_filepath=self.dataset_cfg.annotation_filepaths.train,
                auxiliary_dicts=self.dataset_cfg.auxiliary_dicts.train,
                labels=self.dataset_cfg.labels,
                text_template=self.dataset_cfg.text_template
            )

            self.validate = self.dataset_cls(
                image_dir=self.dataset_cfg.image_dirs.validate,
                annotation_filepath=self.dataset_cfg.annotation_filepaths.validate,
                auxiliary_dicts=self.dataset_cfg.auxiliary_dicts.validate,
                labels=self.dataset_cfg.labels,
                text_template=self.dataset_cfg.text_template
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = self.dataset_cls(
                image_dir=self.dataset_cfg.image_dirs.test,
                annotation_filepath=self.dataset_cfg.annotation_filepaths.test,
                auxiliary_dicts=self.dataset_cfg.auxiliary_dicts.test,
                labels=self.dataset_cfg.labels,
                text_template=self.dataset_cfg.text_template
            )

        if stage == "predict" or stage is None:
            self.predict = self.dataset_cls(
                image_dir=self.dataset_cfg.image_dirs.predict,
                annotation_filepath=self.dataset_cfg.annotation_filepaths.predict,
                auxiliary_dicts=self.dataset_cfg.auxiliary_dicts.predict,
                labels=self.dataset_cfg.labels,
                text_template=self.dataset_cfg.text_template
            )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)