import os
import tqdm
import pickle as pkl
import numpy as np
import pandas as pd

from typing import List
from torch.utils.data import Dataset

class VisionLanguageBase(Dataset):
    def __init__(
        self,
        annotation_filepath: str,
        image_dir: str,
        labels: List[str]
    ):
        self.image_dir = image_dir
        self.labels = labels

        # Discard all -1 labels. These labels are placeholder for None
        annotations = pd.read_json(annotation_filepath, lines=True)
        for l in labels:
            annotations = annotations[annotations[l] != -1]
        
        self.annotations = annotations.reset_index()

    def __len__(self):
        return len(self.annotations)


class LanguageBase(Dataset):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        input_template: str,
        output_template: str,
        label2word: dict,
        labels: List[str]
    ):  
        self.annotations = pd.read_json(annotation_filepath, lines=True)
        self.auxiliary_data = self._load_auxiliary(auxiliary_dicts)
        self.labels = labels

        self.input_template = input_template
        self.output_template = output_template
        self.label2word = label2word

    def _load_auxiliary(self, auxiliary_dicts: dict):
        data = {}
        for key, filepath in auxiliary_dicts.items():
            with open(filepath, "rb") as f:
                data[key] = pkl.load(f)

        return data

    def __len__(self):
        return len(self.annotations)