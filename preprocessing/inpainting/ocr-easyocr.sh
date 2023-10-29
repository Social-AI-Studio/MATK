# IMAGE_DIR=/mnt/data1/datasets/memes/fhm/images/img/
# INPAINT_DIR=/mnt/data1/datasets/memes/fhm/inpainting-files/

# python3 easy-ocr.py \
#     --image-dir /mnt/data1/datasets/memes/fhm/images/img/ \
#     --output-dir /mnt/data1/datasets/memes/fhm/inpainting-files/

# python3 easy-ocr.py \
#     --image-dir /mnt/data1/datasets/memes/harmeme/images/ \
#     --output-dir /mnt/data1/datasets/memes/harmeme/inpainting-files/

# python3 easy-ocr.py \
#     --image-dir /mnt/data1/datasets/memes/mami/images/img/train \
#     --output-dir /mnt/data1/datasets/memes/mami/inpainting-files/

# python3 easy-ocr.py \
#     --image-dir /mnt/data1/datasets/memes/mami/images/img/trial \
#     --output-dir /mnt/data1/datasets/memes/mami/inpainting-files-trial/

# python3 easy-ocr.py \
#     --image-dir /mnt/data1/datasets/memes/mami/images/img/test \
#     --output-dir /mnt/data1/datasets/memes/mami/inpainting-files-test/

# python3 generate-mask.py \
#     --ocr-box-filepath /mnt/data1/datasets/memes/fhm/inpainting-files/easy_ocr.box.json \
#     --image-dir /mnt/data1/datasets/memes/fhm/images/img/ \
#     --output-dir /mnt/data1/datasets/memes/fhm/images/mask/easy-ocr

# python3 generate-mask.py \
#     --ocr-box-filepath /mnt/data1/datasets/memes/harmeme/inpainting-files/easy_ocr.box.json \
#     --image-dir /mnt/data1/datasets/memes/harmeme/images/ \
#     --output-dir /mnt/data1/datasets/memes/harmeme/images/mask/easy-ocr

# python3 generate-mask.py \
#     --ocr-box-filepath /mnt/data1/datasets/memes/mami/inpainting-files/easy_ocr.box.json \
#     --image-dir /mnt/data1/datasets/memes/mami/images/img/train \
#     --output-dir /mnt/data1/datasets/memes/mami/images/mask/train/easy-ocr

# python3 generate-mask.py \
#     --ocr-box-filepath /mnt/data1/datasets/memes/mami/inpainting-files-trial/easy_ocr.box.json \
#     --image-dir /mnt/data1/datasets/memes/mami/images/img/trial \
#     --output-dir /mnt/data1/datasets/memes/mami/images/mask/trial/easy-ocr

python3 generate-mask.py \
    --ocr-box-filepath /mnt/data1/datasets/memes/mami/inpainting-files-test/easy_ocr.box.json \
    --image-dir /mnt/data1/datasets/memes/mami/images/img/test \
    --output-dir /mnt/data1/datasets/memes/mami/images/mask/test/easy-ocr


# python3 deepfillv2-inpainting.py \
#     --image-dir /mnt/data1/datasets/memes/harmeme/images/img \
#     --mask-dir /mnt/data1/datasets/memes/harmeme/images/mask/easy-ocr \
#     --output-dir /mnt/data1/datasets/memes/harmeme/images/easy-ocr-clean
