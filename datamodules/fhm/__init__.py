
import lightning.pytorch as pl

from typing import Optional
from functools import partial

from transformers import AutoProcessor, AutoTokenizer
from torch.utils.data import DataLoader

from . import challenge, finegrained
from ..utils import image_collate_fn, text_collate_fn

class VisionLanguageDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(
        self,
        annotation_filepaths: dict,
        image_dir: str,
        model_class_or_path: str,
        batch_size: int,
        shuffle_train: bool,
        dataset_type: str,
        task: str
    ):
        super().__init__()

        self.annotation_filepaths = annotation_filepaths
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.task = task

        processor = AutoProcessor.from_pretrained(model_class_or_path)
        self.collate_fn = partial(image_collate_fn, processor=processor)

        if dataset_type == "finegrained":
            self.dataset_class = finegrained.VisionLanguageDataset
        elif dataset_type == "challenge":
            self.dataset_class = challenge.VisionLanguageDataset
        else:
            raise NotImplementedError(f"'{dataset_type}' not implemented...")

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = self.dataset_class(
                annotation_filepath=self.annotation_filepaths["train"],
                image_dir=self.image_dir,
                task=self.task
            )

            self.validate = self.dataset_class(
                annotation_filepath=self.annotation_filepaths["validate"],
                image_dir=self.image_dir,
                task=self.task
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = self.dataset_class(
                annotation_filepath=self.annotation_filepaths["test"],
                image_dir=self.image_dir,
                task=self.task
            )

        if stage == "predict" or stage is None:
            self.predict = self.dataset_class(
                annotation_filepath=self.annotation_filepaths["predict"],
                image_dir=self.image_dir,
                task=self.task
            )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)


class LanguageDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(
        self,
        annotation_filepaths: dict,
        model_class_or_path: str,
        auxiliary_dicts: dict,
        input_template: str,
        output_template: str,
        label2word: dict,
        batch_size: int,
        shuffle_train: bool,
        dataset_type: str,
        task: str
    ):
        super().__init__()

        # ensure that word for each label is a single token.
        tokenizer = AutoTokenizer.from_pretrained(model_class_or_path, use_fast=False)
        for word in label2word.values():
            encoded = tokenizer.encode(word, add_special_tokens=False)
            assert len(encoded) == 1

        self.annotation_filepaths = annotation_filepaths
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.auxiliary_dicts = auxiliary_dicts
        self.input_template = input_template
        self.output_template = output_template
        self.label2word = label2word
        self.task = task
        self.collate_fn = partial(text_collate_fn, tokenizer=tokenizer)

        if dataset_type == "finegrained":
            self.dataset_class = finegrained.LanguageDataset
        elif dataset_type == "challenge":
            self.dataset_class = challenge.LanguageDataset
        else:
            raise NotImplementedError(f"'{dataset_type}' not implemented...")

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = self.dataset_class(
                annotation_filepath=self.annotation_filepaths["train"],
                auxiliary_dicts=self.auxiliary_dicts,
                input_template=self.input_template,
                output_template=self.output_template,
                label2word=self.label2word,
                task=self.task
            )

            self.validate = self.dataset_class(
                annotation_filepath=self.annotation_filepaths["validate"],
                auxiliary_dicts=self.auxiliary_dicts,
                input_template=self.input_template,
                output_template=self.output_template,
                label2word=self.label2word,
                task=self.task
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = self.dataset_class(
                annotation_filepath=self.annotation_filepaths["test"],
                auxiliary_dicts=self.auxiliary_dicts,
                input_template=self.input_template,
                output_template=self.output_template,
                label2word=self.label2word,
                task=self.task
            )

        if stage == "predict" or stage is None:
            self.predict = self.dataset_class(
                annotation_filepath=self.annotation_filepaths["predict"],
                auxiliary_dicts=self.auxiliary_dicts,
                input_template=self.input_template,
                output_template=self.output_template,
                label2word=self.label2word,
                task=self.task
            )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)
