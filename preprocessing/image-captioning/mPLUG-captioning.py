import os
import sys
import json
import tqdm
import torch
import numpy as np
import argparse

from PIL import Image
from transformers import AutoTokenizer
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

def yield_partition(image_files, num_partitions, partition_idx):
    partitions = np.array_split(image_files, num_partitions)
    selected_partition = partitions[partition_idx]
    print(f"Partition Index: {partition_idx}")
    print(f"Num. Records: {len(selected_partition)}")

    return selected_partition

def load_model(pretrained_ckpt):
    model = MplugOwlForConditionalGeneration.from_pretrained(
        pretrained_ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
    processor = MplugOwlProcessor(image_processor, tokenizer)

    return model, tokenizer, processor

def main(pretrained_ckpt, image_dir, output_dir, num_partitions, partition_idx):
    image_files = os.listdir(image_dir)
    image_files = yield_partition(image_files, num_partitions, partition_idx)

    os.makedirs(output_dir, exist_ok=True)

    # loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
    # this also loads the associated image processors
    print("Loading model...")
    model, tokenizer, processor = load_model(pretrained_ckpt)

    print("Performing model inference...")
    instruction = "The following is a conversation between a curious human and AI assistant." \
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
    prompt = "Write a short description for the image."
    conv = [
        f'''{instruction}
        Human: <image>
        Human: {prompt}
        AI: '''
    ]

    # generate kwargs (the same in transformers) can be passed in the do_generate()
    generate_kwargs = {
        'do_sample': False,
        'no_repeat_ngram_size': 2,
        'max_new_tokens': 128
    }

    for image_filename in tqdm.tqdm(image_files):

        # Check if it exists
        image_name, _ = os.path.splitext(image_filename)
        output_filepath = os.path.join(output_dir, f"{image_name}.json")
        if os.path.exists(output_filepath):
            continue


        images = [Image.open(os.path.join(image_dir, image_filename)).convert("RGB")]
        
        # preprocess the image
        inputs = processor(text=conv, images=images, return_tensors='pt')
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # generate caption
        with torch.no_grad():
            res = model.generate(**inputs, **generate_kwargs)
        sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)

        record = {
            "img": image_filename,
            "caption": sentence
        }
    
        image_name, _ = os.path.splitext(image_filename)
        output_filepath = os.path.join(output_dir, f"{image_name}.json")
        with open(output_filepath, "w") as f:
            f.write(json.dumps(record))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Perform Image Captioning")
    parser.add_argument("--pretrained-ckpt", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-partitions", type=int, required=True)
    parser.add_argument("--partition-idx", type=int, required=True)
    args = parser.parse_args()

    main(args.pretrained_ckpt, args.image_dir, args.output_dir, args.num_partitions, args.partition_idx)