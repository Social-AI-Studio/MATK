import tqdm
from torch.utils.data import DataLoader

from typing import Optional
from functools import partial
from .collators.text import text_collate_fn
from transformers import AutoTokenizer
from .utils import import_class, ConcatDataset

import lightning.pytorch as pl


class TextDataModule(pl.LightningDataModule):
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
        self.dataset_cfg_len = len(self.dataset_cfg)
        print("Dataset cfg len is "+str(self.dataset_cfg_len))
        print("Multitask adjusted batch size is "+str(self.batch_size//self.dataset_cfg_len))

        # Initialise tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_class_or_path)
        self.tokenizer_class_or_path = tokenizer_class_or_path

        labels = []
        for dataset in dataset_cfg:
            # ensure that word for each label is a single token.
            for label2word in dataset_cfg[dataset].labels.values():
                for _, word in label2word.items():
                    encoded = tokenizer.encode(word, add_special_tokens=False)
                    assert len(encoded) == 1

            # Import the individual dataset classes
            dataset_cfg[dataset].dataset_class = import_class(
                dataset_cfg[dataset].dataset_class
            )

            # Retrieve labels
            dataset_labels = list(dataset_cfg[dataset].labels.keys())
            labels.extend(dataset_labels)

        # Partially load the collate functions
        self.collate_fn = partial(
            text_collate_fn, tokenizer, labels=labels)

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
                    labels_template=cfg.labels_template,
                    labels_mapping=cfg.labels
                )
                self.train.append(dataset_obj)

                dataset_obj = cfg.dataset_class(
                    annotation_filepath=cfg.annotation_filepaths.validate,
                    auxiliary_dicts=cfg.auxiliary_dicts.validate,
                    text_template=cfg.text_template,
                    labels_template=cfg.labels_template,
                    labels_mapping=cfg.labels
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
                    labels_template=cfg.labels_template,
                    labels_mapping=cfg.labels
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
                    labels_template=cfg.labels_template,
                    labels_mapping=cfg.labels
                )
                self.predict.append(dataset_obj)

    def train_dataloader(self):
        return DataLoader(ConcatDataset(*self.train), batch_size=self.batch_size//self.dataset_cfg_len, num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return DataLoader(ConcatDataset(*self.validate), batch_size=self.batch_size//self.dataset_cfg_len, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(ConcatDataset(*self.test), batch_size=self.batch_size//self.dataset_cfg_len, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(ConcatDataset(*self.predict), batch_size=self.batch_size//self.dataset_cfg_len, num_workers=self.num_workers, collate_fn=self.collate_fn)
