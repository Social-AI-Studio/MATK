Meme Preprocessing
===================

In this subfolder, we have implemented the popular techniques that helps with:

#. Text removal using image in-painting techniques
    * keras_ocr to detect text, numpy for masking, and opencv for inpainting
    * mmediting (the open-source image editing toolbox) for inpainting
#. Latent representation extraction using *image captioning* techniques

.. contents:: Table of Contents 
   :depth: 2


*****************
Image In-Painting
*****************

Requirements
------------

keras-ocr
~~~~~~~~~

Pull Tensorflow Docker Image, a latest version (Updated as of March 2023) is nvcr.io/nvidia/tensorflow:21.09-tf2-py3.

.. code-block:: bash

    docker pull nvcr.io/nvidia/tensorflow:21.09-tf2-py3


Next, create (run) a docker container using the docker image.

.. code-block:: bash

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


After we run the container, install the necessary packages:

.. code-block:: bash

    pip install keras-ocr
    apt-get update
    apt-get install python3-opencv -y
    pip install -U tensorflow


mmediting
~~~~~~~~~

Install PyTorch following official instructions. Then, install MMCV with MIM.

.. code-block:: bash

    pip3 install openmim
    mim install mmcv-full 

Next, clone repo and install:

.. code-block:: bash

    git clone https://github.com/HimariO/mmediting-meme.git
    cd mmediting-meme
    pip install mmcv-full==1.1.1+torch1.6.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
    pip install cython --no-cache-dir -e .
    apt update && apt install -y libgl1-mesa-glx


Download ocr.py and Pre-trained Model `DeepFillV2 <https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_places_20200619-10d15793.pth>`_ and save in mmediting-meme folder. 


Usage
------------

keras-ocr
~~~~~~~~~

.. code-block:: bash
    
    python3 inpainting \
        --img-dir <image-dir> \
        --output-dir <output-dir> \

mmediting
~~~~~~~~~

Let data be path to you images folder. Create and move your images to img folder such that the new source image path would be data\img. Later after running, the cleaned images will be in img_cleaned folder in data.
cd to where you have ocr.py and run the followings. Remember to replace  with actual path.

**Step 1**. Detect

.. code-block:: bash

    python3 ocr.py detect <data>


**Step 2**. Convert point annotation to box

.. code-block:: bash

    python3 ocr.py point_to_box <data>/ocr.json


**Step 3**. Create text segmentation mask

.. code-block:: bash

    python3 ocr.py generate_mask <data>/ocr.box.json <data>/img <data>/img_mask_3px

**Step 4**. Inpainting

.. code-block:: bash

    python3 demo/inpainting_demo.py \
    configs/inpainting/deepfillv2/deepfillv2_256x256_8x2_places.py \
    deepfillv2_256x256_8x2_places_20200619-10d15793.pth \
    <data>/img_mask_3px/ <data>/img_cleaned








****************
Image Extraction
****************


Supported Models
----------------
  
**CLIP**

* `https://github.com/openai/CLIP`_ 

**Faster R-CNN**

* `https://github.com/eladsegal/gqa_lxmert/blob/main/notebook.ipynb`_


Requirements
------------

CLIP
~~~~~~~~~

To use clip, please install the *clip* library using pip. 

Faster R-CNN
~~~~~~~~~~~~~~~~~~~~~

None

Execution Instructions
----------------------

Here, we provided command line templates that executes the corresponding feature extraction techniques. Note that you are required to change the `<placeholder>` variables accordingly.`

CLIP
~~~~~~~~~

.. code-block:: bash

    python3 clip-captioning.py \
        --clean-img-dir <clean-img-dir> \
        --model-dir <model-dir> \
        --output-dir <output-dir> \
        --device <device>


Faster R-CNN
~~~~~~~~~~~~

.. code-block:: bash

    python3 clip_features.py \
        --model-dir <model-dir> \
        --device <device>
        --image-dir <image-dir> \
        --feature-dir <feature-dir> \
        


****************
Image Captioning
****************


Supported Models
----------------
**Captioning Model**

*  `BLIP-2 <https://github.com/salesforce/LAVIS>`_
  
**Large Vision-Language Model/Assistant**

* `mPLUG-Owl <https://github.com/X-PLUG/mPLUG-Owl>`_ :sup:`1,2`
* `InstructBLIP <https://github.com/salesforce/LAVIS>`_ :sup:`2`

Requirements
------------

mPLUG-Owl
~~~~~~~~~

To use mPLUG-Owl vision-language assistant, you are required to install the packages mentioned in the `mPLUG-Owl <https://github.com/X-PLUG/mPLUG-Owl>`_ repositry. 

BLIP-2 & InstructBLIP
~~~~~~~~~~~~~~~~~~~~~

To use BLIP-2 or InstructBLIP for image captioning, you are required to install `LAVIS <https://github.com/salesforce/LAVIS>`_ library. 

**Note**: InstructBLIP is currently not supported by the PyPI installation. Hence, you will need to re-install LAVIS using source installation.

OFA
~~~~~~~

To use OFA-Sys for image captioning, you are required to run the following commands:

.. code-block:: bash
    
    git clone --single-branch --branch feature/add_transformers https://github.com/OFA-Sys/OFA.git
    pip install OFA/transformers/
    git lfs install
    git clone https://huggingface.co/OFA-Sys/OFA-Huge


Vision-Language Assistant Set-Up
--------------------------------

You can customize the behaviour of the AI assistant using specific guidelines/instructions. 

#. The instruction can be modified via the ``INSTRUCTION`` variable in the scripts.
#. The prompt message can be modified via the ``PROMPT`` variable in the scripts


Execution Instructions
----------------------

Here, we provided command line templates that executes the corresponding captioning techniques. Note that you are required to change the `<placeholder>` variables accordingly.`

mPLUG-Owl
~~~~~~~~~

.. code-block:: bash

    python3 mPLUG-captioning.py \
        --pretrained-ckpt MAGAer13/mplug-owl-llama-7b \
        --image-dir <image-dir> \
        --output-dir <output-dir> \
        --num-partitions 1 \
        --partition-idx 0


InstructBLIP
~~~~~~~~~~~~

.. code-block:: bash

    python3 InstructBLIP-captioning.py \
        --model-name blip2_vicuna_instruct \
        --model-type vicuna7b \
        --image-dir <image-dir> \
        --output-dir <output-dir> \
        --device cuda \
        --num-partitions 1 \
        --partition-idx 0
