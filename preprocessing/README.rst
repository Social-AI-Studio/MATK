Meme Preprocessing
===================

In this subfolder, we have implemented the popular techniques that helps with: 
#. Text removal using *image in-painting* techniques
#. Visual features extraction using *image captioning* techniques

.. contents:: Table of Contents 
   :depth: 2


*****************
Image In-Painting
*****************

Coming Soon...

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