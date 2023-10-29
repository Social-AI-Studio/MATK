import os
import tqdm
import json
import shutil
import argparse
import numpy as np

from PIL import Image
from skimage import transform

import matplotlib.pyplot as plt
from multiprocessing import Pool

def multi_boxes_mask(image, boxes, pad_crop=5):
    """
    image: np.uint8 (h, w, c)
    boxes: np.int32 (n, 4) ymin, xmin, ymax, xmax
    """
    image = image.copy()
    mask = np.zeros_like(image)
    ih, iw, _ = image.shape
    
    for box in boxes:
        box[:2] = np.maximum(box[:2] - pad_crop, 0)
        box[2:] = np.minimum(box[2:] + pad_crop, image.shape[:2])
        
        patch = image[box[0]: box[2], box[1]: box[3], :]
        pure_white = (patch > 253).all(axis=-1).astype(np.uint8)
        mask[box[0]: box[2], box[1]: box[3], :] = pure_white[..., None]
    
    shift = 4
    shifts = [
        (0, 0), (shift, 0), (-shift, 0), (0, shift), (0, -shift),
        (shift, shift), (-shift, shift), (shift, -shift), (-shift, -shift)
    ]
    # shifts = []
    for offset in shifts:
        ox, oy = offset
        _mask = mask.copy()

        _mask = _mask[
            max(0, 0 + oy): min(ih, ih + oy),
            max(0, 0 + ox): min(iw, iw + ox),
            :
        ]
        crop_pad = [
            (max(0, -oy), max(0, oy)),
            (max(0, -ox), max(0, ox)),
            (0, 0)
        ]
        _mask = np.pad(_mask, crop_pad)
        mask = np.clip(_mask + mask, 0, 1)

    image = image * (1 - mask)
    mask *= 255
    return image, mask

def _mask_white_txt(args):
    img_name, img_boxes, img_dir, out_dir = args
    img_path = os.path.join(img_dir, img_name)

    out_path, _ = os.path.splitext(img_name)
    out_path = os.path.join(out_dir, f"{out_path}.png")
    
    # if os.path.exists(out_path):
    #     return
    
    img = np.array(Image.open(img_path).convert('RGB'))
    img_boxes = [box_info[0] for box_info in img_boxes]
    if len(img_boxes) > 0:
        boxes = np.asarray(img_boxes, dtype=np.int32)
        # print(boxes)
        boxes = np.concatenate([boxes[:, ::-1][:, 2:], boxes[:,::-1][:, :2]], axis=1)
        # print(boxes)
        # x,y,x,y -> y,x,y,x
        # res = inpaint_model.inpaint_multi_boxes(img, boxes)
        masked_img, mask = multi_boxes_mask(img, boxes)

        Image.fromarray(masked_img).save(out_path)
    else:
        print("??")
        shutil.copy(img_path, out_path)
        mask = np.zeros_like(img)

    out_path, _ = os.path.splitext(img_name)
    out_path = os.path.join(out_dir, f"{out_path}.mask4.png")
    Image.fromarray(mask).save(out_path)

def draw_bboxes(img, img_boxes):
    if len(img_boxes) > 0:

    else:
        mask = np.zeros_like(img)


def main(ocr_box_filepath, img_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(ocr_box_filepath, 'r') as f:
        boxes_anno = json.load(f)
    
    boxes_anno = {k:v for k, v in boxes_anno.items() if "1500" in k}
    with Pool(16) as pool:
        args = [
            (img_name, img_boxes, image_dir, output_dir)
            
        ]

    for img_name, img_boxes in tqdm.tqdm(boxes_anno.items()):
        img_path = os.path.join(img_dir, img_name)
        img = np.array(Image.open(img_path).convert('RGB'))
        
        img_boxes = [box_info[0] for box_info in img_boxes]
        boxes = np.asarray(img_boxes, dtype=np.int32)
        boxes = np.concatenate([boxes[:, ::-1][:, 2:], boxes[:,::-1][:, :2]], axis=1)

        # Sanity Check
        img_with_bboxes = draw_bboxes(img, img_boxes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--ocr-box-filepath', type=str, required=True)
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    # calculate square
    main(args.ocr_box_filepath, args.image_dir, args.output_dir)