import os
import tqdm
import numpy as np
import pickle as pkl

from PIL import Image
from . import utils

from typing import List
from torch.utils.data import Dataset

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

class FHMFGBase(Dataset):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        labels: List[str]
    ):
        self.annotations = self._preprocess_annotations(annotation_filepath)
        self.auxiliary_data = self._load_auxiliary(auxiliary_dicts)
        self.labels = labels

    def _preprocess_annotations(self, annotation_filepath: str):
        annotations = []

        # load the default annotations
        data = utils._load_jsonl(annotation_filepath)

        # translate labels into numeric values
        for record in tqdm.tqdm(data, desc="Preprocessing labels"):
            record["img"] = os.path.basename(record["img"])

            if "gold_hate" in record:
                record["hate"] = HATEFULNESS[record["gold_hate"][0]]

            if "gold_pc" in record:
                record["pc"] = [PROTECTED_CATEGORY[x] for x in record["gold_pc"]]

            if "gold_attack" in record:
                record["attack"] = [PROTECTED_ATTACK[x] for x in record["gold_attack"]]

            annotations.append(record)
        
        return annotations

    def _load_auxiliary(self, auxiliary_dicts: dict):
        data = {}
        for key, filepath in tqdm.tqdm(auxiliary_dicts.items(), desc="Loading auxiliary info"):
            with open(filepath, "rb") as f:
                data[key] = pkl.load(f)

        return data

    def __len__(self):
        return len(self.annotations)



class FasterRCNNDataset(FHMFGBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        labels: List[str],
        feats_dict: dict
    ):
        super().__init__(annotation_filepath, auxiliary_dicts, labels)
        self.feats_dict = feats_dict

    def __getitem__(self, idx: int):
        record = self.annotations[idx]

        text = record['text']
        image_id = record['img']
        id, _ = os.path.splitext(image_id)

        item = {
            'id': id,
            'image_id': image_id,
            'text': text,
            'roi_features': self.feats_dict[id]['roi_features'],
            'normalized_boxes': self.feats_dict[id]['normalized_boxes']
        }

        for l in self.labels:
            item[l] = record[l]

        return item


class ImagesDataset(FHMFGBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        labels: List[str],
        image_dir: str
    ):
        super().__init__(annotation_filepath, auxiliary_dicts, labels)
        self.image_dir = image_dir

    def __getitem__(self, idx: int):
        record = self.annotations[idx]

        image_filename = record['img']
        image_id, _ = os.path.splitext(image_filename)

        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image = image.convert("RGB") if image.mode != "RGB" else image

        item = {
            'id': record['id'],
            'image_id': image_id,
            'text': record['text'],
            'image': np.array(image),
            'image_path': image_path
        }

        for l in self.labels:
            item[l] = record[l]

        return item


class TextDataset(FHMFGBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        labels: List[str],
        input_template: str,
        output_template: str,
        label2word: dict
    ):
        super().__init__(annotation_filepath, auxiliary_dicts, labels)
        self.input_template = input_template
        self.output_template = output_template
        self.label2word = label2word

    def __getitem__(self, idx: int):
        record = self.annotations[idx]

        # Format the input template
        input_kwargs = {"text": record['text']}
        for key, data in self.auxiliary_data.items():
            input_kwargs[key] = data[f"{id:05}"]

        image_id, _ = os.path.splitext(record['img'])

        item = {
            'id': record["id"],
            'image_id': image_id,
            'text': self.input_template.format(**input_kwargs)
        }

        for l in self.labels:
            label = record[l]
            item[l] = self.output_template.format(label=self.label2word[label])

        return item