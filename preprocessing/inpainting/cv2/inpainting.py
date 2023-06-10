import keras_ocr
import cv2
import math
import numpy as np
import os
from os import listdir
import multiprocessing

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

def inpaint_text(img_path):
    # read the image 
    try:
        img = keras_ocr.tools.read(img_path) 
        # Recogize text (and corresponding regions)
        # Each list of predictions in prediction_groups is a list of
        # (word, box) tuples. 
        pipeline = keras_ocr.pipeline.Pipeline()
        prediction_groups = pipeline.recognize([img])
        #Define the mask for inpainting
        mask = np.zeros(img.shape[:2], dtype="uint8")
        rec = 0
        #If keras-ocr cannot recognize, then only prints "processing image" error and skip image
        for box in prediction_groups[0]:
            rec = 1 #in case pipeline not recognize
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
            inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
        if(rec):
            return(inpainted_img)
    except:
        print(f'Error reading image at: {img_path}')

def process_image(image_path, cleaned_dir):
    # Check if the image has already been processed
    image_name = os.path.basename(image_path)
    if image_name in os.listdir(cleaned_dir):
        return
    cleaned_img = inpaint_text(image_path)
    # Save the cleaned image
    try:    
        cv2.imwrite(f'{cleaned_dir}/{image_name}', cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB))
    except:
        print(f'Error processing image: {image_name}')

def process_images(img_dir, cleaned_dir):
    # Get a list of image paths to process
    image_type = ('.png','.jpg','.jpeg', '.bmp', '.jpe', '.PNG', '.JPG', '.JPEG', '.JPE', '.BMP')
    image_paths = [os.path.join(img_dir, image) for image in os.listdir(img_dir) if image.endswith(image_type)]
    # Create a multiprocessing pool
    pool = multiprocessing.Pool() #number of processes = multiprocessing.cpu_count()

    # Process the images in parallel
    pool.starmap(process_image, [(image_path, cleaned_dir) for image_path in image_paths])

    # Close the pool
    pool.close()
    pool.join()

if __name__ == '__main__':
    img_dir = "path/to/source/image/directory"
    cleaned_dir = "path/to/destination/inpainted/image/directory"
    process_images(img_dir, cleaned_dir)
