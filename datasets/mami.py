import os
import tqdm
import numpy as np
import pickle as pkl

from PIL import Image
from . import utils

from typing import List
from torch.utils.data import Dataset



class MamiBase(Dataset):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        labels: List[str]
    ):
        self.labels = labels
        self.annotations = self._preprocess_annotations(annotation_filepath)
        self.auxiliary_data = self._load_auxiliary(auxiliary_dicts)
        

    def _preprocess_annotations(self, annotation_filepath: str):
        annotations = []

        # load the default annotations
        data = utils._load_jsonl(annotation_filepath)

        ## handle id stuff
        
        # translate labels into numeric values
        record_id = 0
        for record in tqdm.tqdm(data, desc="Preprocessing labels"):
            record["img"] = record.pop("file_name")
            record["text"] = record.pop("Text Transcription")
            record["id"] = record_id

            if "misogynous" in self.labels:
                record["misogynous"] = int(record.pop("misogynous"))
            else:
                record["shaming"] = int(record.pop("shaming"))
                record["objectification"] = int(record.pop("objectification"))
                record["violence"] = int(record.pop("violence"))
                record["stereotype"] = int(record.pop("stereotype"))

            record_id+=1
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

class FRCNNDataset(MamiBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        labels: List[str],
        text_template: str,
        image_dir: str,
        feats_dir: dict
    ):
        super().__init__(annotation_filepath, auxiliary_dicts, labels)
        self.text_template = text_template
        self.image_dir = image_dir
        self.feats_dict = self._load_feats(feats_dir) if feats_dir != None else None

    def _load_feats(self, feats_dir: str):
        data = {}
        for record in tqdm.tqdm(self.annotations, desc="Loading FRCNN features"):
            image_filename = record['img']

            filename, _ = os.path.splitext(image_filename)
            filepath = os.path.join(feats_dir, f"{filename}.pkl")
            with open(filepath, "rb") as f:
                data[image_filename] = pkl.load(f)

        return data

    def __getitem__(self, idx: int):
        record = self.annotations[idx]

        image_filename = record['img']
        id, _ = os.path.splitext(image_filename)

        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path)
        image = image.convert("RGB") if image.mode != "RGB" else image

        # text formatting
        input_kwargs = {"text": record['text']}
        for key, data in self.auxiliary_data.items():
            input_kwargs[key] = data[image_filename]
        text = self.text_template.format(**input_kwargs)

        item = {
            'id': id,
            'image_id': image_filename,
            'image': np.array(image),
            'image_path': image_path,
            'text': text
        }

        if self.feats_dict:
            item['roi_features'] = self.feats_dict[image_filename]['roi_features']
            item['normalized_boxes'] = self.feats_dict[image_filename]['normalized_boxes']

        for l in self.labels:
            item[l] = record[l]

        return item


class ImageDataset(MamiBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        text_template: str,
        labels: List[str],
        image_dir: str
    ):
        super().__init__(annotation_filepath, auxiliary_dicts, labels)
        self.image_dir = image_dir
        self.text_template = text_template

    def __getitem__(self, idx: int):
        record = self.annotations[idx]

        image_filename = record['img']

        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image = image.convert("RGB") if image.mode != "RGB" else image

        # text formatting
        input_kwargs = {"text": record['text']}
        for key, data in self.auxiliary_data.items():
            input_kwargs[key] = data[image_filename]
        text = self.text_template.format(**input_kwargs)

        item = {
            'id': record['id'],
            'image_filename': image_filename,
            'text': text,
            'image': np.array(image),
            'image_path': image_path
        }

        for l in self.labels:
            item[l] = record[l]

        return item

class TextClassificationDataset(MamiBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        text_template: str,
        output_template: str,
        cls_labels: dict
    ):
        super().__init__(annotation_filepath, auxiliary_dicts, list(cls_labels.keys()))
        self.text_template = text_template
        self.output_template = output_template
        self.cls_labels = cls_labels

    def __getitem__(self, idx: int):
        record = self.annotations[idx]

        # Format the input template
        input_kwargs = {"text": record['text']}
        for key, data in self.auxiliary_data.items():
            input_kwargs[key] = data[f"{id:05}"]
        text = self.text_template.format(**input_kwargs)

        item = {
            'id': record["id"],
            'image_id': record['img'],
            'text': text
        }

        for cls_name, label2word in self.cls_labels.items():
            label = record[cls_name]
            item[cls_name] = self.output_template.format(label=label2word[label])

        return item