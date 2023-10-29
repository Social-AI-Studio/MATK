import os
import tqdm
import random
import numpy as np
import pickle as pkl

from PIL import Image
from typing import List

from . import utils
from .base import CommonBase

from transformers import AutoTokenizer

# binary classification
HATEFULNESS = {
    v:k for k,v in enumerate([
        "not_hateful",
        "hateful"
    ])
}

# 6-class multi-label classification
PROTECTED_CATEGORY = {
    v:k for k,v in enumerate([
        "pc_empty",
        "disability",
        "nationality",
        "race",
        "religion",
        "sex"
    ])
}

# 8-class multi-label classification
PROTECTED_ATTACK = {
    v:k for k,v in enumerate([
        "attack_empty",
        "contempt",
        "dehumanizing",
        "exclusion",
        "inciting_violence",
        "inferiority",
        "mocking",
        "slurs"
    ])
}

DATASET_PREFIX = "fhm_finegrained"

class FHMFGBase(CommonBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        tokenizer_class_or_path: str,
        text_template: str,
        labels_template: str,
        labels_mapping: List[str]
    ):
        super().__init__()
        self.annotations = utils._load_jsonl(annotation_filepath)
        self.auxiliary_data = self._load_auxiliary(auxiliary_dicts)
        self.annotations = self.annotations[:32]
    
        tokenizer = None
        if tokenizer_class_or_path:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_class_or_path)

        self._preprocess_inputs(
            tokenizer,
            text_template
        )
        self._preprocess_labels(
            tokenizer,
            labels_template,
            labels_mapping
        )

        del tokenizer

    def _preprocess_inputs(
            self, 
            tokenizer: AutoTokenizer,
            text_template: str
        ):
        for record in tqdm.tqdm(self.annotations, desc="Input Preprocessing"):
            record["img"] = os.path.basename(record["img"])
            record["id"] = os.path.splitext(record["img"])[0]

            if tokenizer:
                # preprocess text
                input_kwargs = {"text": record['text']}
                for key, data in self.auxiliary_data.items():
                    input_kwargs[key] = data[record["id"]]
                text = text_template.format(**input_kwargs)
                record["formatted_text"] = text

                # perform tokenization
                tokenized = tokenizer(text, return_tensors="pt")
                record["input_ids"] = tokenized.input_ids.squeeze(0)
                record["attention_mask"] = tokenized.attention_mask.squeeze(0)

                if "token_type_ids" in tokenized:
                    record["token_type_ids"] = tokenized.token_type_ids.squeeze(0)


    def _preprocess_labels(
            self, 
            tokenizer: AutoTokenizer, 
            labels_template: str,
            labels_mapping: dict
        ):
        for record in tqdm.tqdm(self.annotations, desc="Preprocessing labels"):
            # convert label to numeric values
            record[f"{DATASET_PREFIX}_hate"] = HATEFULNESS[record["gold_hate"][0]]

            # tokenize labels
            if tokenizer and labels_template:
                key = f"{DATASET_PREFIX}_hate"
                label_text = labels_mapping[key][record[key]]
                tokenized = tokenizer(label_text, return_tensors="pt")
                record[f"{DATASET_PREFIX}_hate_input_ids"] = tokenized.input_ids.squeeze(0)
                record[f"{DATASET_PREFIX}_hate_attention_mask"] = tokenized.attention_mask.squeeze(0)

    def __len__(self):
        return len(self.annotations)

class FRCNNDataset(FHMFGBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        tokenizer_class_or_path: str,
        text_template: str,
        img_dir: str,
        feats_dir: dict
    ):
        super().__init__(
            annotation_filepath, 
            auxiliary_dicts, 
            tokenizer_class_or_path,
            text_template,
            None,
            None
        )
        if feats_dir:
            self.feats_dict = self._load_feats(self.annotations, feats_dir) 
        else:
            self.img_dict = self._load_images(self.annotations, img_dir)

    def __getitem__(self, idx: int):
        record = self.annotations[idx]
        record_id = record['img']

        # Load image or image features
        if self.feats_dict:
            record['roi_features'] = self.feats_dict[record_id]['roi_features']
            record['normalized_boxes'] = self.feats_dict[record_id]['normalized_boxes']
        else:
            record['img'] = self.img_dict[record_id]

        return record


class ImageDataset(FHMFGBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        tokenizer_class_or_path: str,
        text_template: str,
        img_dir: str
    ):
        super().__init__(
            annotation_filepath, 
            auxiliary_dicts, 
            tokenizer_class_or_path,
            text_template,
            None,
            None
        )
        self.image_dict = self._load_images(self.annotations, img_dir)

    def __getitem__(self, idx: int):
        record = self.annotations[idx]
        record['img'] = self.image_dict[record['id']]
        return self.annotations[idx]

class TextDataset(FHMFGBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        tokenizer_class_or_path: str,
        text_template: str,
        labels_template: str,
        labels_mapping: dict
    ):
        super().__init__(
            annotation_filepath, 
            auxiliary_dicts,
            tokenizer_class_or_path,
            text_template,
            labels_template,
            labels_mapping
        )

    def __getitem__(self, idx: int):
        return self.annotations[idx]