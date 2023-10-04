import os
import math
import tqdm
import pickle
import argparse
from typing import List

import torch

from gqa_lxmert.processing_image import Preprocess
from gqa_lxmert.modeling_frcnn import GeneralizedRCNN
from gqa_lxmert.lxmert_utils import Config

def extract_features(
        frcnn_class_or_path: str,
        image_dir: str,
        feature_dir: str,
        device: str
    ):
    device = torch.device(device)

    # load models and model components
    frcnn_cfg = Config.from_pretrained(frcnn_class_or_path)
    frcnn = GeneralizedRCNN.from_pretrained(frcnn_class_or_path, config=frcnn_cfg).to(device)
    image_preprocess = Preprocess(frcnn_cfg)

    filenames = os.listdir(image_dir)
    feature_dir = os.path.join(feature_dir, frcnn_class_or_path)
    os.makedirs(feature_dir, exist_ok=True)
    
    # run frcnn
    for filename in tqdm.tqdm(filenames, desc="Extracting features"):
        filepath = os.path.join(image_dir, filename)
        images, sizes, scales_yx = image_preprocess(filepath)

        output_dict = frcnn(
            images.to(device),
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections=frcnn_cfg.max_detections,
            return_tensors="pt",
        )

        filename, _ = os.path.splitext(filename)
        feature_filepath = os.path.join(feature_dir, f"{filename}.pkl")
        with open(feature_filepath, "wb+") as f:
            pickle.dump(output_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extracting features from memes images")
    parser.add_argument("--frcnn_class_or_path", help="Faster-RCNN model name or model filepath", default="unc-nlp/frcnn-vg-finetuned")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], required=True)
    parser.add_argument("--image-dir", help="input directory containing the meme images", required=True)
    parser.add_argument("--feature-dir", help="output directory for the extracted features", required=True)
    args = parser.parse_args()

    extract_features(
        args.frcnn_class_or_path,
        args.image_dir,
        args.feature_dir,
        args.device
    )