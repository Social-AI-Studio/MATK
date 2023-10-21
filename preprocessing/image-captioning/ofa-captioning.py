import os
import json
import tqdm
import argparse

from PIL import Image
from torchvision import transforms
from transformers import OFAModel, OFATokenizer

def main(
    img_dir: str,
    model_dir: str,
    output_dir: str,
    device: str,
    overwrite: bool
):
    # load img preprocessing function
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    resolution = 480
    patch_resize_transform = transforms.Compose([
        lambda img: img.convert("RGB"),
        transforms.Resize((resolution, resolution),
                          interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # load the tokenizer
    tokenizer = OFATokenizer.from_pretrained(model_dir)
    model = OFAModel.from_pretrained(model_dir, use_cache=False)
    model.to(device)

    # prepare the input tokens
    txt = " what does the img describe?"
    inputs = tokenizer([txt], return_tensors="pt").input_ids

    # create folder if not exists
    output_dir = os.path.join(output_dir, model_dir)
    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(img_dir)
    for filename in tqdm.tqdm(files, desc=f"Running {model_dir}"):
        # check if file exists
        img_name, _ = os.path.splitext(filename)
        output_filepath = os.path.join(output_dir, f"{img_name}.json")
        if os.path.exists(output_filepath) and not overwrite:
            continue

        # load and prepare the img
        img_path = os.path.join(img_dir, filename)
        img = Image.open(img_path)
        patch_img = patch_resize_transform(img).unsqueeze(0).to(device)

        # generate caption
        gen = model.generate(
            inputs.to(device), patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3)
        caption = tokenizer.batch_decode(
            gen, skip_special_tokens=True)[0].strip()

        # save record
        record = {
            "img": filename,
            "caption": caption
        }

        with open(output_filepath, "w+") as f:
            f.write(json.dumps(record))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--img-dir', type=str, required=True)
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument("--device", type=str,
                        choices=["cuda", "cpu"], required=True)
    parser.add_argument('--overwrite', action="store_true")

    # parse arguments
    args = parser.parse_args()

    main(args.img_dir, args.model_dir,
         args.output_dir, args.device, args.overwrite)
