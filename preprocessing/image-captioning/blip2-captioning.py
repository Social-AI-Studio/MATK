import os
import json
import tqdm
import torch
import argparse

from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

def main(model_name, image_dir, output_dir, device):
    image_files = os.listdir(image_dir)

    device = torch.device(device)

    # loads BLIP caption model
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, device_map=device)

    model_name = os.path.basename(model_name) 
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    print("Performing model inference...")
    for filename in tqdm.tqdm(image_files):

        # Check if it exists
        image_name, _ = os.path.splitext(filename)
        output_filepath = os.path.join(output_dir, f"{image_name}.json")
        if os.path.exists(output_filepath):
            continue

        image = Image.open(os.path.join(image_dir, filename))
        image = image.convert("RGB")
        
        # preprocess the image
        inputs = processor(image, return_tensors="pt").to(device)

        # generate caption
        # caption = model.generate({"image": image})
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        record = {
            "img": filename,
            "caption": caption
        }

        with open(output_filepath, "w") as f:
            f.write(json.dumps(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Perform Image Captioning")
    parser.add_argument("--model-name", type=str, default="Salesforce/blip2-opt-6.7b-coco")
    parser.add_argument("--img-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, required=True, choices=["cpu", "cuda"])
    args = parser.parse_args()

    main(args.model_name, args.img_dir, args.output_dir, args.device)