from datamodules.fhm import FHMDataset, image_collate_fn
from models.flava import FlavaClassificationModel

from typing import Optional
from transformers import FlavaProcessor

from torch.utils.data import DataLoader
from functools import partial

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import torch
import numpy as np

from argparse import ArgumentParser
from utils.args import (
    add_trainer_args,
    add_train_args,
    add_test_args
)


class FHMDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, model_class_or_path, batch_size, shuffle_train):
        super().__init__()

        self.annotations_fp = {
            "train": "/mnt/sda/datasets/mmf/datasets/hateful_memes/defaults/annotations/train.jsonl",
            "validate": "/mnt/sda/datasets/mmf/datasets/hateful_memes/defaults/annotations/dev_seen.jsonl",
            "test": "/mnt/sda/datasets/mmf/datasets/hateful_memes/defaults/annotations/dev_seen.jsonl",
            "predict": "/mnt/sda/datasets/mmf/datasets/hateful_memes/defaults/annotations/dev_seen.jsonl",
        }

        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.img_dir = "/mnt/sda/datasets/mmf/datasets/hateful_memes/defaults/images/img/"

        processor = FlavaProcessor.from_pretrained(model_class_or_path)
        self.collate_fn = partial(image_collate_fn, processor=processor)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = FHMDataset(
                annotations_file=self.annotations_fp["train"],
                img_dir=self.img_dir
            )

            self.validate = FHMDataset(
                annotations_file=self.annotations_fp["validate"],
                img_dir=self.img_dir
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = FHMDataset(
                annotations_file=self.annotations_fp["test"],
                img_dir=self.img_dir
            )

        if stage == "predict" or stage is None:
            self.predict = FHMDataset(
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


def main(args):
    # Reproducibility
    pl.seed_everything(args.seed, workers=True)

    # Initialize the FlavaForSequenceClassification model
    model = FlavaClassificationModel("facebook/flava-full")

    # Initialize the Datasets
    dataset = FHMDataModule("facebook/flava-full",
                            args.batch_size, args.shuffle_train)

    # callbacks
    callbacks = []
    chkpt_callback = ModelCheckpoint(
        dirpath="checkpoints/flava_fhm/",
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        every_n_epochs=1,
        save_last=True,
    )
    callbacks.append(chkpt_callback)

    if args.early_stopping:
        es_callback = EarlyStopping(
            monitor="val_auroc",
            patience=3,
            mode="max"
        )
        callbacks.append(es_callback)

    # Training Model
    if args.do_train:
        trainer = pl.Trainer(
            accelerator=args.accelerator,
            devices=args.devices,
            max_epochs=args.num_epochs,
            accumulate_grad_batches=args.accumulate_gradients,
            callbacks=callbacks,
            strategy='ddp_find_unused_parameters_true',
            limit_train_batches=1
        )

        trainer.fit(model, dataset)
    
    if args.do_test:
        chkpt_filepath = args.saved_model_filepath if args.saved_model_filepath else chkpt_callback.best_model_path
        model = FlavaClassificationModel.load_from_checkpoint(chkpt_filepath)
        print(f"Loaded model checkpoint: {chkpt_filepath}")

        trainer = pl.Trainer(
            accelerator=args.accelerator,
            devices=1,
            max_epochs=args.num_epochs,
            callbacks=callbacks,
            strategy='ddp_find_unused_parameters_true'
        )

        trainer.test(model, dataset)
    
    if args.do_predict:

        chkpt_filepath = args.saved_model_filepath if args.saved_model_filepath else chkpt_callback.best_model_path
        model = FlavaClassificationModel.load_from_checkpoint(chkpt_filepath)
        print(f"Loaded model checkpoint: {chkpt_filepath}")

        trainer = pl.Trainer(
            accelerator=args.accelerator,
            devices=1,
            max_epochs=args.num_epochs,
            callbacks=callbacks,
            strategy='ddp_find_unused_parameters_true'
        )

        results = trainer.predict(model, dataset)

        preds, labels = [], []
        for d in results:
            preds.append(d['preds'])
            labels.append(d['labels'])

        preds = torch.cat(preds).argmax(dim=-1)
        labels = torch.cat(labels)
        results = torch.stack((preds, labels), dim=1).numpy()
        results = results.astype(int)

        result_filepath = args.saved_model_filepath.replace(".ckpt", ".csv")
        np.savetxt(result_filepath,
                   results, fmt='%d', delimiter=',')


if __name__ == "__main__":
    parser = ArgumentParser()
    add_trainer_args(parser)
    add_train_args(parser)
    add_test_args(parser)

    args = parser.parse_args()

    main(args)
