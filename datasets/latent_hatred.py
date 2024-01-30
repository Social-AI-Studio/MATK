import os
import tqdm
import numpy as np
import pickle as pkl

from PIL import Image
from . import utils

from typing import List
from .base import CommonBase

DATASET_PREFIX = "latent_hatred"

class LatentHatredBase(CommonBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        text_template: str,
        labels_template: str,
        labels_mapping: List[str]
    ):
        super().__init__()
        self.annotations = utils._load_jsonl(annotation_filepath)
        # keys: post, target, implied_statement

        self._preprocess_dataset()
        self.auxiliary_data = self._load_auxiliary(auxiliary_dicts)

        self._format_input_output(
            text_template,
            labels_template,
            labels_mapping
        )

    def _preprocess_dataset(self):
        for record in tqdm.tqdm(self.annotations, desc="Dataset preprocessing"):
            record['text'] = record['post']
            record[f"{DATASET_PREFIX}_label"] = record['class']
            # record['latent_hatred_implied_statement'] = record['implied_statement']

    def _format_input_output(
        self,
        text_template: str,
        labels_template: str,
        labels_mapping: dict
    ):
        for record in tqdm.tqdm(self.annotations, desc="Input/Output formatting"):
            # format input text template
            input_kwargs = {"text": record["post"]}
            for key, data in self.auxiliary_data.items():
                input_kwargs[key] = data[record["id"]]
            text = text_template.format(**input_kwargs)
            record["templated_text"] = text

            # format output text template (for text-to-text generation)
            if labels_mapping:
                for cls_name, label2word in labels_mapping.items():
                    label = record[cls_name]
                    record[f"templated_{cls_name}"] = labels_template.format(
                        label=label2word[label] + ". " + record["generated_implied_statement"]
                    )

    def __len__(self):
        return len(self.annotations)

class TextDataset(LatentHatredBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        text_template: str,
        labels_template: str,
        labels_mapping: dict
    ):
        super().__init__(
            annotation_filepath,
            auxiliary_dicts,
            text_template,
            labels_template,
            labels_mapping
        )

    def __getitem__(self, idx: int):
        return self.annotations[idx]