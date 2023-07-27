import os
import sys
import json
import tqdm
import torch
import numpy as np
import argparse

from PIL import Image
from lavis.models import load_model_and_preprocess

def yield_partition(image_files, num_partitions, partition_idx):
    partitions = np.array_split(image_files, num_partitions)
    selected_partition = partitions[partition_idx]
    print(f"Partition Index: {partition_idx}")
    print(f"Num. Records: {len(selected_partition)}")

    return selected_partition

def main(model_name, model_type, image_dir, output_dir, device, num_partitions, partition_idx):
    image_files = os.listdir(image_dir)
    image_files = yield_partition(image_files, num_partitions, partition_idx)

    device = torch.device(device)

    os.makedirs(output_dir, exist_ok=True)

    # loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
    # this also loads the associated image processors
    print("Loading model...")
    model, vis_processors, _ = load_model_and_preprocess(name=model_name, model_type=model_type, is_eval=True, device=device)

    print("Performing model inference...")
    for image_filename in tqdm.tqdm(image_files):
        image = Image.open(os.path.join(image_dir, image_filename))
        image = image.convert("RGB")
        
        # preprocess the image
        image = vis_processors["eval"](image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image, "prompt": "Write a short description for the image."})

        record = {
            "img": image_filename,
            "caption": caption[0]
        }

        image_name, _ = os.path.splitext(image_filename)
        output_filepath = os.path.join(output_dir, f"{image_name}.json")
        with open(output_filepath, "w") as f:
            f.write(json.dumps(record))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Perform Image Captioning")
    parser.add_argument("--model-name", type=str, required=True, choices=["blip2_vicuna_instruct", "blip2_t5_instruct"])
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--device", type=str, required=True, choices=["cpu", "cuda"])
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-partitions", type=int, required=True)
    parser.add_argument("--partition-idx", type=int, required=True)
    args = parser.parse_args()

    main(args.model_name, args.model_type, args.image_dir, args.output_dir, args.device, args.num_partitions, args.partition_idx)