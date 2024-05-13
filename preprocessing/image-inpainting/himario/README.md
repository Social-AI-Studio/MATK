# mmediting

Inpainting using open-source image editing toolbox [mmediting](https://github.com/open-mmlab/mmediting).
## Installation

Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/). Then, install MMCV with MIM. 
```bash
pip3 install openmim
mim install mmcv-full 
```
Next, clone repo and install:

```bash
git clone https://github.com/HimariO/mmediting-meme.git
cd mmediting-meme
pip install mmcv-full==1.1.1+torch1.6.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
pip install cython --no-cache-dir -e .
apt update && apt install -y libgl1-mesa-glx
```
Download ```ocr.py``` and Pre-trained Model [DeepFillV2](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_places_20200619-10d15793.pth) and save in ```mmediting-meme``` folder.



## Usage
Let ```data``` be path to you images folder. Create and move your images to ```img``` folder such that the new source image path would be ```data\img```. Later after running, the cleaned images will be in ```img_cleaned``` folder in ```data```.

```cd``` to where you have ```ocr.py``` and run the followings. Remember to replace __<data>__ with actual path.

__Step 1.__ Detect
```bash
python3 ocr.py detect <data>
```
__Step 2.__ Convert point annotation to box
```bash
python3 ocr.py point_to_box <data>/ocr.json
```
__Step 3.__ Create text segmentation mask
```bash
python3 ocr.py generate_mask <data>/ocr.box.json <data>/img <data>/img_mask_3px
```
__Step 4.__ Inpainting

```bash
python3 demo/inpainting_demo.py \
configs/inpainting/deepfillv2/deepfillv2_256x256_8x2_places.py \
deepfillv2_256x256_8x2_places_20200619-10d15793.pth \
 <data>/img_mask_3px/ <data>/img_cleaned
```
## Acknowledgment

This follows [HimariO](https://github.com/HimariO/HatefulMemesChallenge)'s preprocessing code in his winning solution for Facebook Hateful Memes Challenge.

