import os
import sys
import json
import tqdm
import torch
import argparse

from PIL import Image
from lavis.models import load_model_and_preprocess

def main(model_name, model_type, image_dir, output_dir, device):
    device = torch.device(device)
    image_files = os.listdir(image_dir)

    os.makedirs(output_dir, exist_ok=True)

    # loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
    # this also loads the associated image processors
    model, vis_processors, _ = load_model_and_preprocess(name=model_name, model_type=model_type, is_eval=True, device=device)

    records = []
    for image_filename in tqdm.tqdm(image_files):
        image = Image.open(os.path.join(image_dir, image_filename))
        image = image.convert("RGB")
        
        # preprocess the image
        image = vis_processors["eval"](image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image, "prompt": "Write a short description for the image."})

        records.append({
            "img": image_filename,
            "caption": caption[0]
        })
    
    output_filepath = os.path.join(output_dir, f"{model_name}-{model_type}.jsonl")
    with open(output_filepath, "w") as f:
        for item in records:
            f.write(json.dumps(item))
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Perform Image Captioning")
    parser.add_argument("--model-name", type=str, required=True, choices=["blip2_vicuna_instruct", "blip2_t5_instruct"])
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--device", type=str, required=True, choices=["cpu", "cuda"])
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    main(args.model_name, args.model_type, args.image_dir, args.output_dir, args.device)