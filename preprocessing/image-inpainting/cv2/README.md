# Inpainting.py
This Python file takes as input 2 paths: ```img_dir``` as path to source image directory and ```cleaned_dir``` as path to empty destination directory. It will use __keras_ocr__ to detect text, __numpy__ for masking, and __opencv__ for inpainting. Inpainted images will be saved in ```img_dir```. __multiprocessing__ is used to inpaint many images in parallel thus saves computing time.
## Installation
Run the following commands in terminal:
### Create a Tensorflow Docker Container
Pull Tensorflow Docker Image, a latest version (Updated as of March 2023) is [nvcr.io/nvidia/tensorflow:21.09-tf2-py3](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow/tags).
 

```bash
docker pull nvcr.io/nvidia/tensorflow:21.09-tf2-py3
```
You can ensure that the docker image has been downloaded via 
```bash
docker images
```
Next, create (run) a docker container using the docker image.
```bash
#replace <> with corresponding arguments
docker run -it -p <local_port>:<container_port> --name <docker_name> --shm-size 10G --gpus all -v <external_dir>:/mnt/sda/<username> nvcr.io/nvidia/tensorflow:21.09-tf2-py3 bash

# Useful Argument Options
# --gpus <0, 1 or all>  (This depicts the GPU resources that your docker can see)
# -i  (Keep STDIN open even if not attached)
# -t  (Allocate a pseudo-tty)
# -p <local_port>:<container_port>  (Allow you to access applications on specific ports, e.g. Jupyter Notebook)
# -v <local_dir>:<container_dir> (Mount a local directory into docker directory)

# Example Docker Container
docker run -it -p 8808:8808 --name nguyen-tensorflow  --shm-size 10G --gpus all -v /mnt/sda/nguyen_hoang:/mnt/sda/nguyen_hoang nvcr.io/nvidia/tensorflow:21.09-tf2-py3 bash
```

### Install keras-ocr and opencv

After we run the container, install the necessary packages:
```bash
pip install keras-ocr
apt-get update
apt-get install python3-opencv -y
pip install -U tensorflow
```
## Usage
In ```__main__``` of ```inpainting.py``` change the corresponding paths, then saves.
```bash
if __name__ == '__main__':
    img_dir = "path/to/source/image/directory"
    cleaned_dir = "path/to/destination/inpainted/image/directory"
    process_images(img_dir, cleaned_dir)

```

 ```cd``` to the folder containing ```inpainting.py``` and run:
```bash
python3 inpainting.py
```
or run in background as larger datasets will take a long time:
```bash
nohup python3 inpainting.py &
```
You can check progress with:
```bash
tail -f nohup.out
```