import os
import cv2
import math
import json
import argparse
import numpy as np

from PIL import Image

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

def get_mask(img, ocr_results):
    # define inpainting mask
    mask = np.zeros(img.shape[:2], dtype="uint8")
    
    #If keras-ocr cannot recognize, then only prints "processing image" error and skip image
    for box in ocr_results:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 

        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)

        #For the line thickness, we will calculate the length of the line between 
        #the top-left corner and the bottom-left corner.
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))

        #Define the line and inpaint
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)

    return mask

def main(img_dir, mask_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(img_dir)
    for img_filename in files:
        # load image
        img_filepath = os.path.join(img_dir, img_filename)
        img = np.array(Image.open(img_filepath))
        
        image_filename = os.path.basename(img_filename)
        mask_filename, _ = os.path.splitext(image_filename)
        mask_filename = f"{mask_filename}.mask.png"
        mask_filepath = os.path.join(mask_dir, mask_filename)
        mask = np.array(Image.open(mask_filepath)).astype(np.uint8)
        mask = mask[:, :, 0]
        print(mask.shape, img.shape)
        inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
        
        # mask_filename, _ = os.path.splitext(img_filename)
        # mask_filename = os.path.join("/mnt/data1/datasets/memes/mami/images/mask/test/", f"{mask_filename}.mask.png")
        # cv2.imwrite(mask_filename, cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        
        output_filename = os.path.join(output_dir, img_filename)
        cv2.imwrite(output_filename, cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB))
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--img-dir', type=str,required=True)
    parser.add_argument('--mask-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    main(args.img_dir, args.mask_dir, args.output_dir)
