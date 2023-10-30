import os
import json
import tqdm
import numpy as np
import pickle as pkl

from PIL import Image
from typing import List
from torch.utils.data import Dataset

class CommonBase(Dataset):
    def _load_auxiliary(self, auxiliary_dicts: dict):
        data = {}
        for key, aux_dir in auxiliary_dicts.items():
            data[key] = {}
            for record in tqdm.tqdm(self.annotations, desc="Loading auxiliary info"):
                filepath = os.path.join(aux_dir, f"{record['id']}.json")
                with open(filepath, "r") as f:
                    d = json.load(f)[key]
                    d = d.strip()
                    d = d[: -1] if d[-1] == "." else d
                    data[key][record['id']] = d

        return data

    def _load_images(self, annotations: List[object], img_dir: str):
        image_dict = {}
        for record in tqdm.tqdm(annotations, desc="Loading images"):
            # image loading and processing
            image_path = os.path.join(img_dir, record['img'])
            image = Image.open(image_path)
            image = image.resize((224, 224))
            image = image.convert("RGB") if image.mode != "RGB" else image
            image_dict[record['id']] = np.array(image)

        return image_dict

    def _load_feats(self, annotations: List[object], feats_dir: str):
        data = {}
        for record in tqdm.tqdm(annotations, desc="Loading FRCNN features"):
            image_filename = record['img']

            filename, _ = os.path.splitext(image_filename)
            filepath = os.path.join(feats_dir, f"{filename}.pkl")
            with open(filepath, "rb") as f:
                data[image_filename] = pkl.load(f)

        return data

    def __len__(self):
        return len(self.annotations)
