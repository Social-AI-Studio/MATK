import os
import json
import shutil
import argparse
import numpy as np

from PIL import Image
from skimage import transform

import matplotlib.pyplot as plt
from multiprocessing import Pool

def is_light_gray_or_white(img, threshold=245):
    img_sample = img.copy()

    # Check if all three channels are close to each other (considered light gray)
    # new_axis = np.expand_dims(img_sample[:, :, 0], axis=-1)
    # img_sample = np.concatenate((img_sample, new_axis), axis=-1)
    # img_diff = np.diff(img_sample, axis=-1)

    # diff_criteria = np.all(np.abs(img_diff) < 250, axis=-1)
    mean = np.expand_dims(np.mean(img_sample, axis=-1), axis=-1)
    img_diff = np.diff(img_sample - mean, axis=-1)

    diff_criteria = np.all(np.abs(img_diff) < 5, axis=-1)
    
    # Check if all three channels are above the threshold (considered white)
    threshold_criteria = np.any(img_sample > threshold, axis=-1)

    # return diff_criteria
    return np.logical_or(diff_criteria, threshold_criteria).astype(np.uint8)

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

        # Use numpy.vectorize to apply the function element-wise
        pure_white = is_light_gray_or_white(patch)
        # pure_white = (patch > 253).all(axis=-1).astype(np.uint8)
        # print("HimariO")
        # print(pure_white)
        mask[box[0]: box[2], box[1]: box[3], :] = pure_white[..., None]

    shift = 3
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
        shutil.copy(img_path, out_path)
        mask = np.zeros_like(img)

    out_path, _ = os.path.splitext(img_name)
    out_path = os.path.join(out_dir, f"{out_path}.mask.png")
    Image.fromarray(mask).save(out_path)

def main(ocr_box_filepath, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(ocr_box_filepath, 'r') as f:
        boxes_anno = json.load(f)
    
    # boxes_anno = {k:v for k, v in boxes_anno.items() if "150" in k}
    with Pool(16) as pool:
        args = [
            (img_name, img_boxes, image_dir, output_dir)
            for img_name, img_boxes in boxes_anno.items()
        ]
        pool.map(_mask_white_txt, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--ocr-box-filepath', type=str, required=True)
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    # calculate square
    main(args.ocr_box_filepath, args.image_dir, args.output_dir)