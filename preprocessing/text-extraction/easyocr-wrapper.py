import os
import glob
import json
import tqdm
import numpy as np

import easyocr
import argparse

def cast_pred_type(pred):
    result = []
    for tup in pred:
        coord, txt, score = tup
        coord = np.array(coord).tolist()
        score = float(score)
        result.append((coord, txt, score))
    return result

def detect(img_dir, output_dir):
    reader = easyocr.Reader(['en'])

    images = glob.glob(os.path.join(img_dir, '*.png')) 
    images += glob.glob(os.path.join(img_dir, '**', '*.png')) 
    
    print(len(images))
    assert len(images) > 0 # adjust depending on number of images in image folder

    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, 'ocr.json')
    out_anno = {}
    
    print(f"Find {len(images)} images!")
    for i, image_path in enumerate(tqdm.tqdm(images)):
        img_name = os.path.basename(image_path)
        pred = reader.readtext(image_path)
        pred = cast_pred_type(pred)

        img_filename, _ = os.path.splitext(img_name)
        ocr_filepath = os.path.join(output_dir, f"{img_filename}.json")
        with open(ocr_filepath, "w+") as f:
            json.dump(pred, f)

        out_anno[img_name] = pred

    with open(out_json, 'w') as f:
        json.dump(out_anno, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--img_dir', type=str, help='Path to directory containing cleaned images',required=True)
    parser.add_argument('--output_dir', type=str, help='Path to file with pretrained model weights',required=True)
    
    # parse arguments
    args = parser.parse_args()

    detect(args.img_dir, args.output_dir)
