import torch
import numpy as np
import pandas as pd
import lightning.pytorch as pl

from PIL import Image
from typing import Optional
from functools import partial
from transformers import FlavaProcessor
from torch.utils.data import DataLoader, Dataset

# from datamodules.utils import image_collate_fn
from .utils import image_collate_fn

def get_dataset_attributes(dataset_name: str):
    img_dir = "/mnt/sda/datasets/mmf/datasets/hateful_memes/defaults/images/img/"

    if dataset_name == "fhm":
        return FHMDataset, {
            "train": "/mnt/sda/datasets/mmf/datasets/hateful_memes/defaults/annotations/train.jsonl",
            "validate": "/mnt/sda/datasets/mmf/datasets/hateful_memes/defaults/annotations/dev_seen.jsonl",
            "test": "/mnt/sda/datasets/mmf/datasets/hateful_memes/defaults/annotations/dev_seen.jsonl",
            "predict": "/mnt/sda/datasets/mmf/datasets/hateful_memes/defaults/annotations/dev_seen.jsonl",
        }, img_dir
    elif dataset_name == "fhm_finegrained":
        return FHMFinegrainedDataset, {
            "train": "/mnt/sda/datasets/mmf/datasets/hateful_memes/defaults/annotations/fine_grained/train.jsonl",
            "validate": "/mnt/sda/datasets/mmf/datasets/hateful_memes/defaults/annotations/fine_grained/dev_seen.jsonl",
            "test": "/mnt/sda/datasets/mmf/datasets/hateful_memes/defaults/annotations/fine_grained/dev_seen.jsonl",
            "predict": "/mnt/sda/datasets/mmf/datasets/hateful_memes/defaults/annotations/fine_grained/dev_seen.jsonl",
        }, img_dir

class FHMDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, dataset_name, model_class_or_path, batch_size, shuffle_train, **kwargs):
        super().__init__()

        # TODO: Separate this into a separate YAML configuration file
        self.dataset_class, self.annotations_fp, self.img_dir = get_dataset_attributes(dataset_name)

        self.batch_size = batch_size
        self.shuffle_train = shuffle_train

        processor = FlavaProcessor.from_pretrained(model_class_or_path)
        self.collate_fn = partial(image_collate_fn, processor=processor)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = self.dataset_class(
                annotations_file=self.annotations_fp["train"],
                img_dir=self.img_dir
            )

            self.validate = self.dataset_class(
                annotations_file=self.annotations_fp["validate"],
                img_dir=self.img_dir
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = self.dataset_class(
                annotations_file=self.annotations_fp["test"],
                img_dir=self.img_dir
            )

        if stage == "predict" or stage is None:
            self.predict = self.dataset_class(
                annotations_file=self.annotations_fp["predict"],
                img_dir=self.img_dir
            )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)

class FHMDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_annotations = pd.read_json(annotations_file, lines=True)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_annotations)

    def __getitem__(self, idx):
        img_id = self.img_annotations.loc[idx, 'img']
        text = self.img_annotations.loc[idx, 'text']
        label = self.img_annotations.loc[idx, 'label']
        img_path = self.img_dir + img_id
        
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img = img.convert("RGB") if img.mode != "RGB" else img

        return {
            'id': img_id,
            'text': text, 
            'image': np.array(img),
            'label': label
        }

class FHMFinegrainedDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_annotations = pd.read_json(annotations_file, lines=True)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_annotations)

    def __getitem__(self, idx):
        return self.get_hateful_classification(idx)

    def get_hateful_classification(self, idx):
        id = self.img_annotations.loc[idx, 'id']
        img_id = self.img_annotations.loc[idx, 'img']
        text = self.img_annotations.loc[idx, 'text']
        label = 1 if self.img_annotations.loc[idx, 'gold_hate'] == ["hateful"] else 0
        img_path = self.img_dir + img_id

        img = Image.open(img_path)
        img = img.resize((224, 224))
        img = img.convert("RGB") if img.mode != "RGB" else img

        return {
            'id': img_id,
            'text': text,
            'image': np.array(img),
            'label': label
        }