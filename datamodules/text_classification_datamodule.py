from torch.utils.data import DataLoader

from typing import Optional
from functools import partial
from .collators.text import text_collate_fn
from .utils import import_class
from transformers import AutoTokenizer

import lightning.pytorch as pl

class TextClassificationDataModule(pl.LightningDataModule):
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
        self.num_workers= num_workers
        self.dataset_cls = import_class(dataset_cfg.dataset_class)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_class_or_path)
        self.collate_fn = partial(text_collate_fn, tokenizer=tokenizer, labels=dataset_cfg.labels)

        # ensure that word for each label is a single token.
        for word in dataset_cfg.labels:
            encoded = tokenizer.encode(word, add_special_tokens=False)
            assert len(encoded) == 1

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = self.dataset_cls(
                annotation_filepath=self.dataset_cfg.annotation_filepaths.train,
                auxiliary_dicts=self.dataset_cfg.auxiliary_dicts.train,
                text_template=self.dataset_cfg.text_template,
                output_template=self.dataset_cfg.output_template,
                cls_labels=self.dataset_cfg.cls_labels
            )

            self.validate = self.dataset_cls(
                annotation_filepath=self.dataset_cfg.annotation_filepaths.validate,
                auxiliary_dicts=self.dataset_cfg.auxiliary_dicts.validate,
                text_template=self.dataset_cfg.text_template,
                output_template=self.dataset_cfg.output_template,
                cls_labels=self.dataset_cfg.cls_labels
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = self.dataset_cls(
                annotation_filepath=self.dataset_cfg.annotation_filepaths.test,
                auxiliary_dicts=self.dataset_cfg.auxiliary_dicts.test,
                text_template=self.dataset_cfg.text_template,
                output_template=self.dataset_cfg.output_template,
                cls_labels=self.dataset_cfg.cls_labels
            )

        if stage == "predict" or stage is None:
            self.predict = self.dataset_cls(
                annotation_filepath=self.dataset_cfg.annotation_filepaths.predict,
                auxiliary_dicts=self.dataset_cfg.auxiliary_dicts.predict,
                text_template=self.dataset_cfg.text_template,
                output_template=self.dataset_cfg.output_template,
                cls_labels=self.dataset_cfg.cls_labels
            )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)