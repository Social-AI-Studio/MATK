import os
import glob
import tqdm
import json
import easyocr
import argparse

import numpy as np

def cast_pred_type(pred):
    result = []
    for tup in pred:
        coord, txt, score = tup
        coord = np.array(coord).tolist()
        score = float(score)
        result.append((coord, txt, score))
    return result

def main(image_dir: str, output_dir: str):
    reader = easyocr.Reader(['en'])

    # fetch all images
    images = glob.glob(os.path.join(image_dir, '**')) 
    print(f"Find {len(images)} images!")
    assert len(images) > 0 # adjust depending on number of images in image folder

    # create output_dir
    os.makedirs(output_dir, exist_ok=True)
    ocr = {}
    for image_path in tqdm.tqdm(images, "Performing EasyOCR"):
        img_name = os.path.basename(image_path)
        pred = reader.readtext(image_path)
        ocr[img_name] = cast_pred_type(pred)


    boxed_ocr = {}
    for k, v in ocr.items():
        img_ocr_infos = []
        for txt_info in v:
            coord, txt, score = txt_info
            xmin = min([p[0] for p in coord])
            xmax = max([p[0] for p in coord])
            ymin = min([p[1] for p in coord])
            ymax = max([p[1] for p in coord])
            box = [xmin, ymin, xmax, ymax]
            img_ocr_infos.append([box, txt, score])
        boxed_ocr[k] = img_ocr_infos

    ocr_filepath = os.path.join(output_dir, 'easy_ocr.json')
    with open(ocr_filepath, 'w') as f:
        json.dump(ocr, f)

    box_ocr_filepath = ocr_filepath.replace('.json', '.box.json')
    with open(box_ocr_filepath, 'w') as f:
        json.dump(boxed_ocr, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    # calculate square
    main(args.image_dir, args.output_dir)
    