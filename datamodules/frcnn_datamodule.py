from torch.utils.data import DataLoader
from functools import partial
from typing import Optional
from .utils import import_class, ConcatDataset

import lightning.pytorch as pl
from transformers import AutoTokenizer
from .collators.frcnn import frcnn_collate_fn

from .collators.gqa_lxmert.lxmert_utils import Config
from .collators.gqa_lxmert.processing_image import Preprocess



class FRCNNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_cfg: str,
        tokenizer_class_or_path: str,
        frcnn_class_or_path: str,
        batch_size: int,
        shuffle_train: bool,
        num_workers: int
    ):
        super().__init__()

        self.dataset_cfg = dataset_cfg
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers
        self.tokenizer_class_or_path = tokenizer_class_or_path

        labels = []
        for dataset in self.dataset_cfg:
            
            # Import the individual dataset classes
            self.dataset_cfg[dataset].dataset_class = import_class(
                self.dataset_cfg[dataset].dataset_class
            )

            # Retrieve labels
            labels.extend(self.dataset_cfg[dataset].labels)

        # Partially load the collate functions
        if frcnn_class_or_path != None:
            frcnn_cfg = Config.from_pretrained(frcnn_class_or_path)
            image_preprocess = Preprocess(frcnn_cfg)
        else:
            image_preprocess = None

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_class_or_path)
        self.collate_fn = partial(
            frcnn_collate_fn,
            tokenizer=tokenizer,
            labels=labels,
            image_preprocess=image_preprocess
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = []
            for dataset in self.dataset_cfg:
                cfg = self.dataset_cfg[dataset]
                dataset_obj = cfg.dataset_class(
                    img_dir=cfg.image_dirs.train,
                    annotation_filepath=cfg.annotation_filepaths.train,
                    auxiliary_dicts=cfg.auxiliary_dicts.train,
                    text_template=cfg.text_template,
                    feats_dir=cfg.feats_dir.train
                )
                self.train.append(dataset_obj)

            self.validate = []
            for dataset in self.dataset_cfg:
                cfg = self.dataset_cfg[dataset]

                dataset_obj = cfg.dataset_class(
                    img_dir=cfg.image_dirs.validate,
                    annotation_filepath=cfg.annotation_filepaths.validate,
                    auxiliary_dicts=cfg.auxiliary_dicts.validate,
                    text_template=cfg.text_template,
                    feats_dir=cfg.feats_dir.validate
                )
                self.validate.append(dataset_obj)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = []
            for dataset in self.dataset_cfg:
                cfg = self.dataset_cfg[dataset]
                dataset_obj = cfg.dataset_class(
                    img_dir=cfg.image_dirs.test,
                    annotation_filepath=cfg.annotation_filepaths.test,
                    auxiliary_dicts=cfg.auxiliary_dicts.test,
                    text_template=cfg.text_template,
                    feats_dir=cfg.feats_dir.test
                )
                self.test.append(dataset_obj)

        if stage == "predict" or stage is None:
            self.predict = []
            for dataset in self.dataset_cfg:
                cfg = self.dataset_cfg[dataset]
                dataset_obj = cfg.dataset_class(
                    img_dir=cfg.image_dirs.predict,
                    annotation_filepath=cfg.annotation_filepaths.predict,
                    auxiliary_dicts=cfg.auxiliary_dicts.predict,
                    text_template=cfg.text_template,
                    feats_dir=cfg.feats_dir.predict
                )
                self.predict.append(dataset_obj)

    def train_dataloader(self):  # change the division - make it batch size per dataset
        return DataLoader(ConcatDataset(*self.train), batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return DataLoader(ConcatDataset(*self.validate), batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(ConcatDataset(*self.test), batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(ConcatDataset(*self.predict), batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)
