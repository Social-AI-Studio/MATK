import os
import cv2
import json
import math
import numpy as np
from os import listdir
import multiprocessing
import argparse

from PIL import Image 
  

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

def inpaint_text(img_path, ocr_path):
    # return None
    # read the image 
    try:
        img = cv2.imread(img_path)
        # img = np.array(img).astype('int8')
        with open(ocr_path) as f:
            prediction_groups = json.load(f)

        mask = np.zeros(img.shape[:2], dtype="uint8")
        rec = 0
        #If keras-ocr cannot recognize, then only prints "processing image" error and skip image
        for box in prediction_groups:
            rec = 1 #in case pipeline not recognize
            x0, y0 = box[0][0]
            x1, y1 = box[0][1] 
            x2, y2 = box[0][2]
            x3, y3 = box[0][3] 
            x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
            x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)

            #For the line thickness, we will calculate the length of the line between 
            #the top-left corner and the bottom-left corner.
            thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))

            #Define the line and inpaint
            cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
            thickness)
            inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

        if(rec):
            return rec, mask, inpainted_img
        else: 
            return rec, None, img
    except Exception as e:
        print(e)
        print(f'Error reading image at: {img_path}')

def process_image(image_path, ocr_path, mask_dir, cleaned_dir):
    # Check if the image has already been processed
    image_name = os.path.basename(image_path)
    # if image_name in os.listdir(cleaned_dir):
    #     return
    
    rec, mask, img = inpaint_text(image_path, ocr_path)
    # print(mask)
    # if cleaned_img:
    #     return

    # Save the cleaned image
    try:    
        cv2.imwrite(f'{cleaned_dir}/{image_name}', img)
    except:
        print(f'Unable to save cleaned image for {image_name}!')

    if rec == 0:
        return

    try:    
        names = os.path.splitext(image_name)
        mask_name = f"{names[0]}.mask{names[1]}"
        cv2.imwrite(f'{mask_dir}/{mask_name}', mask)
    except:
        print(f'Unable to save mask for {image_name}!')

def process_images(img_dir, ocr_dir, mask_dir, cleaned_dir):
    # Get a list of image paths to process
    image_type = ('.png','.jpg','.jpeg', '.bmp', '.jpe', '.PNG', '.JPG', '.JPEG', '.JPE', '.BMP')
    image_paths = [os.path.join(img_dir, image) for image in os.listdir(img_dir) if image.endswith(image_type)]
    
    ocr_filename = [f"{os.path.splitext(image)[0]}.json" for image in os.listdir(img_dir) if image.endswith(image_type)]
    ocr_paths = [os.path.join(ocr_dir, filename) for filename in ocr_filename]


    os.makedirs(cleaned_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    # Create a multiprocessing pool
    pool = multiprocessing.Pool() #number of processes = multiprocessing.cpu_count()

    # Process the images in parallel
    pool.starmap(process_image, [(image_path, ocr_path, mask_dir, cleaned_dir) for image_path, ocr_path in zip(image_paths, ocr_paths)])

    # Close the pool
    pool.close()
    pool.join()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--img_dir', type=str, help='Path to directory containing cleaned images',required=True)
    parser.add_argument('--ocr_dir', type=str, help='Path to directory containing cleaned images',required=True)
    parser.add_argument('--mask_dir', type=str, help='Path to file with pretrained model weights',required=True)
    parser.add_argument('--cleaned_dir', type=str, help='Path to file with pretrained model weights',required=True)
    
    # parse arguments
    args = parser.parse_args()

    process_images(args.img_dir, args.ocr_dir, args.mask_dir, args.cleaned_dir)
