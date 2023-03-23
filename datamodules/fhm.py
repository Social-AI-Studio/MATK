import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from PIL import Image

from functools import partial
from torchvision.transforms import ToTensor

from typing import Optional
from transformers import FlavaProcessor

# class FHMDataModule(pl.LightningDataModule):
#     """
#     DataModule used for semantic segmentation in geometric generalization project
#     """

#     def __init__(self, model_class_or_path, batch_size: int = 32):
#         super().__init__()

#         self.annotations_fp = {
#             "train": "/mnt/sda/datasets/mmf/datasets/hateful_memes/defaults/annotations/train.jsonl",
#             "validate": "/mnt/sda/datasets/mmf/datasets/hateful_memes/defaults/annotations/dev_seen.jsonl",
#             "test": "/mnt/sda/datasets/mmf/datasets/hateful_memes/defaults/annotations/dev_seen.jsonl",
#         }

#         self.batch_size = batch_size
#         self.img_dir = "/mnt/sda/datasets/mmf/datasets/hateful_memes/defaults/images/img/"

#         processor = FlavaProcessor.from_pretrained(model_class_or_path)
#         self.collate_fn = partial(image_collate_fn, processor=processor)

#     def setup(self, stage: Optional[str] = None):
#         # if stage == "fit" or stage is None:
#         self.train =  FHMDataset(
#             annotations_file=self.annotations_fp["train"],
#             img_dir=self.img_dir
#         )

#         self.validate = FHMDataset(
#             annotations_file=self.annotations_fp["validate"],
#             img_dir=self.img_dir
#         )

#         # Assign test dataset for use in dataloader(s)
#         # if stage == "test" or stage is None:
#         self.test = FHMDataset(
#             annotations_file=self.annotations_fp["test"],
#             img_dir=self.img_dir
#         )
            
#     def train_dataloader(self):
#         return DataLoader(self.train, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn, shuffle=True)

#     def val_dataloader(self):
#         return DataLoader(self.validate, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)
        

#     def test_dataloader(self):
#         return DataLoader(self.test, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)

def image_collate_fn(batch, processor):
    texts, images = [], []
    for item in batch:
        texts.append(item["text"])
        images.append(item["image"])
    
    inputs = processor(  
        text=texts, images=images, return_tensors="pt", padding=True
    )

    # Get Labels
    label_name = "label"
    if label_name in batch[0].keys():
        labels = [feature[label_name] for feature in batch]
        inputs['labels'] = torch.tensor(labels, dtype=torch.int64)

    return inputs

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