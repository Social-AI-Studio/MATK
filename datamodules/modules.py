import os
import tqdm
import pickle as pkl
import numpy as np


from torch.utils.data import DataLoader
from transformers import AutoTokenizer


# from ..datasets.fhm_finegrained import (
#     TextDataset,
#     ImagesDataset,
#     FasterRCNNDataset
# )
import importlib

from datamodules.collators import get_collator

from typing import List, Optional

import lightning.pytorch as pl

class FasterRCNNDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(
        self,
        annotation_filepaths: dict,
        tokenizer_class_or_path: str,
        dataset_class: str,
        feats_dirs: dict,
        batch_size: int,
        auxiliary_dicts: dict,
        shuffle_train: bool,
        labels: List[str],
        num_workers: int
    ):
        super().__init__()

        self.annotation_filepaths = annotation_filepaths
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.labels = labels
        self.auxiliary_dicts = auxiliary_dicts
        self.num_workers= num_workers
        
        self.feats_dict = {}
        for split in ["train", "validate", "test", "predict"]:
            self.feats_dict[split] = self._load_feats_frcnn(feats_dirs, split)
        
        self.collate_fn = get_collator(
            tokenizer_class_or_path, 
            labels=labels,
        )

        # TEMP HACK
        package_name = ".".join(dataset_class.split(".")[:-1])
        class_name = dataset_class.split(".")[-1]
        m = importlib.import_module(package_name)
        self.dataset_cls = getattr(m, class_name)
    
    def _load_feats_frcnn(self, feats_dirs: str, key: str):
        feats_dict = {}

        files = [
            x for x in os.listdir(feats_dirs[key])
            if ".pkl" in x
        ]
        for filename in tqdm.tqdm(files, desc=f"Loading {key} features"):
            filepath = os.path.join(feats_dirs[key], filename)
            
            filename, _ = os.path.splitext(filename)

            with open(filepath, "rb") as f:
                feats_dict[filename] = pkl.load(f)

        return feats_dict

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["train"],
                auxiliary_dicts=self.auxiliary_dicts,
                feats_dict=self.feats_dict["train"],
                labels=self.labels
            )

            self.validate = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["validate"],
                auxiliary_dicts=self.auxiliary_dicts,
                feats_dict=self.feats_dict["validate"],
                labels=self.labels
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["test"],
                auxiliary_dicts=self.auxiliary_dicts,
                feats_dict=self.feats_dict["test"],
                labels=self.labels
            )

        if stage == "predict" or stage is None:
            self.predict = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["predict"],
                auxiliary_dicts=self.auxiliary_dicts,
                feats_dict=self.feats_dict["predict"],
                labels=self.labels
            )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

class ImagesDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(
        self,
        annotation_filepaths: dict,
        image_dirs: dict,
        tokenizer_class_or_path: str,
        frcnn_class_or_path: str,
        dataset_class: str,
        auxiliary_dicts: dict,
        batch_size: int,
        shuffle_train: bool,
        labels: List[str],
        num_workers: int
    ):
        super().__init__()

        self.annotation_filepaths = annotation_filepaths
        self.image_dirs = image_dirs
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.labels = labels
        self.auxiliary_dicts = auxiliary_dicts
        self.num_workers= num_workers

        self.collate_fn = get_collator(
            tokenizer_class_or_path, 
            labels=labels, 
            frcnn_class_or_path=frcnn_class_or_path
        )

        # TEMP HACK
        package_name = ".".join(dataset_class.split(".")[:-1])
        class_name = dataset_class.split(".")[-1]
        m = importlib.import_module(package_name)
        self.dataset_cls = getattr(m, class_name)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["train"],
                auxiliary_dicts=self.auxiliary_dicts,
                image_dir=self.image_dirs["train"],
                labels=self.labels
            )

            self.validate = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["validate"],
                auxiliary_dicts=self.auxiliary_dicts,
                image_dir=self.image_dirs["validate"],
                labels=self.labels
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["test"],
                auxiliary_dicts=self.auxiliary_dicts,
                image_dir=self.image_dirs["test"],
                labels=self.labels
            )

        if stage == "predict" or stage is None:
            self.predict = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["predict"],
                auxiliary_dicts=self.auxiliary_dicts,
                image_dir=self.image_dirs["predict"],
                labels=self.labels
            )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)


class TextDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(
        self,
        annotation_filepaths: dict,
        tokenizer_class_or_path: str,
        dataset_class: str,
        auxiliary_dicts: dict,
        input_template: str,
        output_template: str,
        label2word: dict,
        batch_size: int,
        shuffle_train: bool,
        labels: List[str],
        num_workers: int
    ):
        super().__init__()

        # ensure that word for each label is a single token.
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_class_or_path, use_fast=False)
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
        self.labels = labels
        self.collate_fn = get_collator(tokenizer_class_or_path, labels=labels)
        self.num_workers= num_workers

        # TEMP HACK
        package_name = ".".join(dataset_class.split(".")[:-1])
        class_name = dataset_class.split(".")[-1]
        m = importlib.import_module(package_name)
        self.dataset_cls = getattr(m, class_name)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["train"],
                auxiliary_dicts=self.auxiliary_dicts,
                input_template=self.input_template,
                output_template=self.output_template,
                label2word=self.label2word,
                labels=self.labels
            )

            self.validate = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["validate"],
                auxiliary_dicts=self.auxiliary_dicts,
                input_template=self.input_template,
                output_template=self.output_template,
                label2word=self.label2word,
                labels=self.labels
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["test"],
                auxiliary_dicts=self.auxiliary_dicts,
                input_template=self.input_template,
                output_template=self.output_template,
                label2word=self.label2word,
                labels=self.labels
            )

        if stage == "predict" or stage is None:
            self.predict = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["predict"],
                auxiliary_dicts=self.auxiliary_dicts,
                input_template=self.input_template,
                output_template=self.output_template,
                label2word=self.label2word,
                labels=self.labels
            )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)
