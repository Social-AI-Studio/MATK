import os
import sys
import torch
import numpy as np
import lightning.pytorch as pl

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from argparse import ArgumentParser
from utils.args import (
    add_trainer_args,
    add_train_args,
    add_test_args
)

from models.flava import FlavaClassificationModel
from datamodules import load_datamodule


def main(args):
    # Reproducibility
    pl.seed_everything(args.seed, workers=True)

    # Initialize the FlavaForSequenceClassification model
    model = FlavaClassificationModel("facebook/flava-full")

    # Initialize the Datasets
    dataset = load_datamodule(args.dataset_name, "facebook/flava-full",
                              batch_size=args.batch_size, shuffle_train=args.shuffle_train)

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
            callbacks=callbacks
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
            callbacks=callbacks
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
