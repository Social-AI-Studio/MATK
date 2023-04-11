import numpy as np

from PIL import Image
from .base import LanguageBase, VisionLanguageBase, Tasks


class VisionLanguageDataset(VisionLanguageBase, Tasks):
    def __init__(
        self,
        annotation_filepath: str,
        image_dir: str,
        task: str
    ):
        super().__init__(annotation_filepath, image_dir, task)

    def get_hateful_cls(self, idx):
        img_id = self.annotations.loc[idx, 'img']
        text = self.annotations.loc[idx, 'text']
        label = self.annotations.loc[idx, 'label']
        img_path = self.image_dir + img_id

        img = Image.open(img_path)
        img = img.resize((224, 224))
        img = img.convert("RGB") if img.mode != "RGB" else img

        return {
            'id': img_id,
            'text': text,
            'image': np.array(img),
            'label': label
        }


class LanguageDataset(LanguageBase, Tasks):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        input_template: str,
        output_template: str,
        label2word: dict,
        task: str
    ):
        super().__init__(annotation_filepath, auxiliary_dicts,
                         input_template, output_template, label2word, task)

    def get_hateful_cls(self, idx):
        id = self.annotations.loc[idx, 'id']
        img_id = self.annotations.loc[idx, 'img']
        text = self.annotations.loc[idx, 'text']
        label = self.annotations.loc[idx, 'label']

        # Format the input template
        input_kwargs = {"text": text}
        for key, data in self.auxiliary_data.items():
            input_kwargs[key] = data[f"{id:05}"]

        return {
            'id': id,
            'text': self.input_template.format(**input_kwargs),
            'label': self.output_template.format(label=self.label2word[label])
        }
