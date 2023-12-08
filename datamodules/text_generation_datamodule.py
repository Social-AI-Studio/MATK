import tqdm
from torch.utils.data import DataLoader

from typing import Optional
from functools import partial
from .collators.text import  text_gen_collate_fn
from transformers import AutoTokenizer
from .utils import import_class, ConcatDataset

import lightning.pytorch as pl


class TextGenerationDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(
        self,
        dataset_cfg: str,
        tokenizer_class_or_path: str,
        batch_size: int,
        shuffle_train: bool,
        num_workers: int
    ):
        super().__init__()

        self.dataset_cfg = dataset_cfg
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers

        # Initialise tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_class_or_path)
        self.tokenizer_class_or_path = tokenizer_class_or_path

        labels = []
        for dataset in dataset_cfg:
            # Import the individual dataset classes
            dataset_cfg[dataset].dataset_class = import_class(
                dataset_cfg[dataset].dataset_class
            )

        # Partially load the collate functions
        self.collate_fn = partial(
            text_gen_collate_fn, tokenizer)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = []
            self.validate = []

            for dataset in self.dataset_cfg:
                cfg = self.dataset_cfg[dataset]
                dataset_obj = cfg.dataset_class(
                    annotation_filepath=cfg.annotation_filepaths.train,
                    auxiliary_dicts=cfg.auxiliary_dicts.train,
                    text_template=cfg.text_template,
                )
                self.train.append(dataset_obj)

                dataset_obj = cfg.dataset_class(
                    annotation_filepath=cfg.annotation_filepaths.validate,
                    auxiliary_dicts=cfg.auxiliary_dicts.validate,
                    text_template=cfg.text_template,
                )
                self.validate.append(dataset_obj)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = []
            for dataset in self.dataset_cfg:
                cfg = self.dataset_cfg[dataset]
                dataset_obj = cfg.dataset_class(
                    annotation_filepath=cfg.annotation_filepaths.test,
                    auxiliary_dicts=cfg.auxiliary_dicts.test,
                    text_template=cfg.text_template,
                )
                self.test.append(dataset_obj)

        if stage == "predict" or stage is None:
            self.predict = []
            for dataset in self.dataset_cfg:
                cfg = self.dataset_cfg[dataset]
                dataset_obj = cfg.dataset_class(
                    annotation_filepath=cfg.annotation_filepaths.predict,
                    auxiliary_dicts=cfg.auxiliary_dicts.predict,
                    text_template=cfg.text_template,
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