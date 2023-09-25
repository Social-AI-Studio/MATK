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
        dataset_prefix: str,
        labels: List[str]
    ):
        self.labels = labels
        self.dataset_prefix = dataset_prefix

    def _encode_labels(self):
        encoded_labels = []
        for label in self.labels:
            new_label = self.dataset_prefix + "_"+label
            encoded_labels.append(new_label)
        return encoded_labels
