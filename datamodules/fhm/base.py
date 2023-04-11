import abc
import pickle as pkl
import pandas as pd

from torch.utils.data import Dataset

class Tasks(abc.ABC):

    @abc.abstractmethod
    def get_hateful_cls(self, idx: int):
        pass

class VisionLanguageBase(Dataset):
    def __init__(
        self,
        annotation_filepath: str,
        image_dir: str,
        task: str
    ):
        self.annotations = pd.read_json(annotation_filepath, lines=True)
        self.image_dir = image_dir
        self.task = task

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int):
        if self.task == "hateful_cls":
            return self.get_hateful_cls(idx)
        else:
            raise NotImplementedError(f"'get_{self.task}' is not defined")


class LanguageBase(Dataset):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        input_template: str,
        output_template: str,
        label2word: dict,
        task: str
    ):  
        self.annotations = pd.read_json(annotation_filepath, lines=True)
        self.auxiliary_data = self._load_auxiliary(auxiliary_dicts)
        self.task = task

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

    def __getitem__(self, idx: int):
        if self.task == "hateful_cls":
            return self.get_hateful_cls(idx)
        else:
            raise NotImplementedError(f"'get_{self.task}' is not defined")