import tqdm
import pickle as pkl

from typing import List
from torch.utils.data import Dataset

class CommonBase(Dataset):
    def __init__(
        self,
        labels: List[str]
    ):
        self.labels = labels

    def _load_auxiliary(self, auxiliary_dicts: dict):
        data = {}
        for key, filepath in tqdm.tqdm(auxiliary_dicts.items(), desc="Loading auxiliary info"):
            with open(filepath, "rb") as f:
                data[key] = pkl.load(f)

        return data

    def __len__(self):
        return len(self.annotations)
