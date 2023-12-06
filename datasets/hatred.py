import os
import tqdm
from . import utils

from typing import List
from .base import CommonBase

DATASET_PREFIX = "hatred"

class HatredBase(CommonBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        text_template: str,
    ):
        super().__init__()
        self.annotations = utils._load_jsonl(annotation_filepath)
        self._preprocess_dataset()
        self.auxiliary_data = self._load_auxiliary(auxiliary_dicts)
        self._format_input_output(
            text_template,
        )

    def _preprocess_dataset(self):
        for record in tqdm.tqdm(self.annotations, desc="Dataset preprocessing"):
            record["img"] = os.path.basename(record["img"])
            record["id"] = os.path.splitext(record["img"])[0]
            record["target"] = record["target"][0]
            ## check?
            record[f"targets"] = record["reasonings"][0]
            

    def _format_input_output(
        self,
        text_template: str,
    ):
        for record in tqdm.tqdm(self.annotations, desc="Input/Output formatting"):
            # format input text template
            input_kwargs = {"text": record['text'], "reasonings": record['reasonings'][0], "target": record["target"]} # check?
            for key, data in self.auxiliary_data.items():
                input_kwargs[key] = data[record["id"]]
            text = text_template.format(**input_kwargs)
            record["templated_text"] = text


    def __len__(self):
        return len(self.annotations)

class FRCNNDataset(HatredBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        text_template: str,
        img_dir: str,
        feats_dir: dict
    ):
        super().__init__(
            annotation_filepath, 
            auxiliary_dicts, 
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

class ImageDataset(HatredBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        text_template: str,
        img_dir: str
    ):
        super().__init__(
            annotation_filepath, 
            auxiliary_dicts, 
            text_template,
            None,
            None
        )
        self.image_dict = self._load_images(self.annotations, img_dir)

    def __getitem__(self, idx: int):
        record = self.annotations[idx]
        record['img'] = self.image_dict[record['id']]
        return self.annotations[idx]

class TextDataset(HatredBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        text_template: str,
    ):
        super().__init__(
            annotation_filepath, 
            auxiliary_dicts,
            text_template,
        )

    def __getitem__(self, idx: int):
        return self.annotations[idx]