import torch
import clip
from PIL import Image
import pickle
import argparse
import os
import tqdm 

def extract_features(model, image_dir, feature_dir, device):
    
    filenames = os.listdir(image_dir)
    feature_dir = os.path.join(feature_dir, "")
    os.makedirs(feature_dir, exist_ok=True)
    
    model, preprocess = clip.load(model, device=device)

    for filename in tqdm.tqdm(filenames, desc="Extracting CLIP features"):
        filepath = os.path.join(image_dir, filename)
        image = preprocess(Image.open(filepath)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)

        filename, _ = os.path.splitext(filename)
        feature_filepath = os.path.join(feature_dir, f"{filename}.pkl")
        with open(feature_filepath, "wb+") as f:
            pickle.dump(image_features, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extracting features from memes images")
    parser.add_argument("--model", help="Model name or filepath", default="ViT-B/32")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], required=True)
    parser.add_argument("--image-dir", help="input directory containing the meme images", required=True)
    parser.add_argument("--feature-dir", help="output directory for the extracted features", required=True)
    args = parser.parse_args()

    extract_features(
        args.model,
        args.image_dir,
        args.feature_dir,
        args.device
    )