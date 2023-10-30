import os
import tqdm

from typing import List
from . import utils
from .base import CommonBase

# binary classification
HATEFULNESS = {
    v: k for k, v in enumerate([
        "not_hateful",
        "hateful"
    ])
}

# 6-class multi-label classification
PROTECTED_CATEGORY = {
    v: k for k, v in enumerate([
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
    v: k for k, v in enumerate([
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
        text_template: str,
        labels_template: str,
        labels_mapping: List[str]
    ):
        super().__init__()
        self.annotations = utils._load_jsonl(annotation_filepath)
        self._preprocess_dataset()

        self.auxiliary_data = self._load_auxiliary(auxiliary_dicts)
        self._format_input_output(
            text_template,
            labels_template,
            labels_mapping
        )

    def _preprocess_dataset(self):
        for record in tqdm.tqdm(self.annotations, desc="Dataset preprocessing"):
            record["img"] = os.path.basename(record["img"])
            record["id"] = os.path.splitext(record["img"])[0]

            # convert label to numeric values
            record[f"{DATASET_PREFIX}_hate"] = HATEFULNESS[record["gold_hate"][0]]

    def _format_input_output(
        self,
        text_template: str,
        labels_template: str,
        labels_mapping: dict
    ):
        for record in tqdm.tqdm(self.annotations, desc="Input/Output formatting"):
            # format input text template
            input_kwargs = {"text": record['text']}
            for key, data in self.auxiliary_data.items():
                input_kwargs[key] = data[record["id"]]
            text = text_template.format(**input_kwargs)
            record["templated_text"] = text

            # format output text template (for text-to-text generation)
            if labels_mapping:
                for cls_name, label2word in labels_mapping.items():
                    label = record[cls_name]
                    record[f"templated_{cls_name}"] = labels_template.format(
                        label=label2word[label]
                    )

    def __len__(self):
        return len(self.annotations)


class FRCNNDataset(FHMFGBase):
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


class ImageDataset(FHMFGBase):
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


class TextDataset(FHMFGBase):
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
