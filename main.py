from data.fhm import FHMDataset, image_collate_fn
from models.flava import FlavaClassificationModel

from typing import Optional
from transformers import FlavaProcessor

from torch.utils.data import DataLoader
from functools import partial

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from argparse import ArgumentParser


class FHMDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, model_class_or_path, batch_size: int = 32):
        super().__init__()

        self.annotations_fp = {
            "train": "/mnt/sda/datasets/mmf/datasets/hateful_memes/defaults/annotations/train.jsonl",
            "validate": "/mnt/sda/datasets/mmf/datasets/hateful_memes/defaults/annotations/dev_seen.jsonl",
            "test": "/mnt/sda/datasets/mmf/datasets/hateful_memes/defaults/annotations/dev_seen.jsonl",
        }

        self.batch_size = batch_size
        self.img_dir = "/mnt/sda/datasets/mmf/datasets/hateful_memes/defaults/images/img/"

        processor = FlavaProcessor.from_pretrained(model_class_or_path)
        self.collate_fn = partial(image_collate_fn, processor=processor)

    def setup(self, stage: Optional[str] = None):
        # if stage == "fit" or stage is None:
        self.train = FHMDataset(
            annotations_file=self.annotations_fp["train"],
            img_dir=self.img_dir
        )

        self.validate = FHMDataset(
            annotations_file=self.annotations_fp["validate"],
            img_dir=self.img_dir
        )

        # Assign test dataset for use in dataloader(s)
        # if stage == "test" or stage is None:
        self.test = FHMDataset(
            annotations_file=self.annotations_fp["test"],
            img_dir=self.img_dir
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)


def main(args):
    # Initialize the FlavaForSequenceClassification model
    model = FlavaClassificationModel("facebook/flava-full")

    # Initialize the Datasets
    dataset = FHMDataModule("facebook/flava-full")

    # callbacks
    chkpt_callback = ModelCheckpoint(
        dirpath="checkpoints/flava_fhm/",
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        every_n_epochs=1,
        save_last=True,
    )
    
    es_callback = EarlyStopping(
        monitor="val_auroc",
        patience=3,
        mode="max"
    )

    # Define the PyTorch Lightning trainer
    trainer = pl.Trainer(
        enable_progress_bar=True,
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.num_epochs,
        strategy='ddp_find_unused_parameters_true',
        callbacks=[chkpt_callback, es_callback],
        accumulate_grad_batches=2,
    )
    trainer.fit(model, dataset)

    print(chkpt_callback.best_model_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    args = parser.parse_args()

    main(args)