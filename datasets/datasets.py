import os
import numpy as np

from PIL import Image
from .base import LanguageBase, VisionLanguageBase

from typing import List

class VLFeaturesDataset(VisionLanguageBase):
    def __init__(
        self,
        annotation_filepath: str,
        labels: List[str],
        feats_dict: dict
    ):
        super().__init__(annotation_filepath, None, labels)
        self.feats_dict = feats_dict

    def __getitem__(self, idx: int):
        text = self.annotations.loc[idx, 'text']
        image_id = self.annotations.loc[idx, 'img']
        id, _ = os.path.splitext(image_id)

        item = {
            'id': id,
            'image_id': image_id,
            'text': text,
            'roi_features': self.feats_dict[id]['roi_features'],
            'normalized_boxes': self.feats_dict[id]['normalized_boxes']
        }

        for l in self.labels:
            item[l] = self.annotations.loc[idx, l]

        return item

class VLImagesDataset(VisionLanguageBase):
    def __init__(
        self,
        annotation_filepath: str,
        image_dir: str,
        labels: List[str]
    ):
        super().__init__(annotation_filepath, image_dir, labels)

    def __getitem__(self, idx: int):
        text = self.annotations.loc[idx, 'text']
        image_id = self.annotations.loc[idx, 'img']
        id, _ = os.path.splitext(image_id)

        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image = image.convert("RGB") if image.mode != "RGB" else image

        item = {
            'id': id,
            'image_id': image_id,
            'text': text,
            'image': np.array(image),
            'image_path': image_path
        }

        for l in self.labels:
            item[l] = self.annotations.loc[idx, l]

        return item


class LanguageDataset(LanguageBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        input_template: str,
        output_template: str,
        label2word: dict,
        labels: List[str]
    ):
        super().__init__(annotation_filepath, auxiliary_dicts,
                         input_template, output_template, label2word, 
                         labels)

    def __getitem__(self, idx: int):
        text = self.annotations.loc[idx, 'text']
        image_id = self.annotations.loc[idx, 'img']
        id, _ = os.path.splitext(image_id)

        # Format the input template
        input_kwargs = {"text": text}
        for key, data in self.auxiliary_data.items():
            input_kwargs[key] = data[f"{id:05}"]

        item = {
            'id': id,
            'image_id': image_id,
            'text': self.input_template.format(**input_kwargs)
        }

        for l in self.labels:
            label = self.annotations.loc[idx, l]
            item[l] = self.output_template.format(label=self.label2word[label])

        return item
