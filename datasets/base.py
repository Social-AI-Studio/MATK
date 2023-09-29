import abc
import os
import pickle as pkl

from PIL import Image
import numpy as np

from typing import List
from torch.utils.data import Dataset

class CommonBase(Dataset):
    def __init__(
        self,
        auxiliary_dicts: dict,
        labels: List[str]
    ):
        # Load auxiliary information and labels
        self.auxiliary_data = self._load_auxiliary(auxiliary_dicts)
        self.labels = labels

    def _load_auxiliary(self, auxiliary_dicts: dict):
        data = {}
        for key, filepath in auxiliary_dicts.items():
            with open(filepath, "rb") as f:
                data[key] = pkl.load(f)

        return data

    def __len__(self):
        return len(self.annotations)


class FeaturesBase(CommonBase):
    def __init__(
        self,
        auxiliary_dicts: dict,
        labels: List[str],
        image_dir: str
    ):
        super().__init__(self, auxiliary_dicts, labels)

        # Unique attributes
        self.image_dir = image_dir


class ImagesBase(CommonBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        labels: List[str],
        image_dir: str
    ):
        super().__init__(self, auxiliary_dicts, labels)

        # Unique attributes
        self.image_dir = image_dir

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


class TextBase(CommonBase, metaclass=abc.ABCMeta):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        labels: List[str],
        input_template: str,
        output_template: str,
        label2word: dict
    ):
        super().__init__(self, auxiliary_dicts, labels)
        self.annotations = self._preprocess_annotations(annotation_filepath)

        # Unique attributes
        self.input_template = input_template
        self.output_template = output_template
        self.label2word = label2word

    @abc.abstractmethod
    def _preprocess_annotations(self, annotation_filepath):
        pass

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
