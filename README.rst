MATK: Meme Analysis ToolKit
===========================

MATK (Meme Analysis Toolkit) aims at training, analyzing and comparing
the state-of-the-art Vision Language Models on the various downstream
memes tasks (i.e. hateful memes classification, attacked group
classification, hateful memes explanation generation).

.. contents:: Table of Contents 
   :depth: 2

***************
Installation
***************

To get started, run the following command::

  pip install -r requirements.txt

***************
Main Features
***************

* Provides a framework for training and evaluating a different multimodal classification models on well known hateful memes datasets
* Allows for efficient experimentation and parameter tuning through modification of configuration files (under configs directory)
* Evaluate models using different state-of-the-art evaluation metrics such as Accuracy and AUROC


***************
Examples and Tutorials
***************

Coming soon...

**************************
Datasets and Preprocessing
**************************


Supported Datasets
~~~~~~~~~~~~~~~~~~
.. |green_check| unicode:: U+2714
   :trim:

+------------------------------+-----------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+------+-------+---------------+
| Dataset                      | Paper                                                           | Source                                                                                                         | Year | Size  | Target/Group  |
+==============================+=================================================================+================================================================================================================+======+=======+===============+
| Facebook Hateful Memes (FHM) | `[arxiv] <https://arxiv.org/pdf/2005.04790.pdf>`_               | `[DrivenData] <https://www.drivendata.org/accounts/login/?next=/competitions/70/hateful-memes-phase-2/data/>`_ | 2020 | 10000 |               |
+------------------------------+-----------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+------+-------+---------------+
| Fine grained FHM             | `[arxiv] <https://aclanthology.org/2021.woah-1.21.pdf>`_        | `[GitHub] <https://github.com/facebookresearch/fine_grained_hateful_memes/tree/main/data>`_                    | 2021 | 10000 | |green_check| |
+------------------------------+-----------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+------+-------+---------------+
| HarMeme                      | `[arxiv] <https://aclanthology.org/2021.findings-acl.246.pdf>`_ | `[GitHub] <https://github.com/di-dimitrov/harmeme>`_                                                           | 2021 | 3544  | |green_check| |
+------------------------------+-----------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+------+-------+---------------+
| Harm-C + Harm-P              | `[arxiv] <https://arxiv.org/pdf/2109.05184v2.pdf>`_             | `[GitHub] <https://github.com/LCS2-IIITD/MOMENTA>`_                                                            | 2021 | 3552  | |green_check| |
+------------------------------+-----------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+------+-------+---------------+
| MAMI                         | `[arxiv] <https://aclanthology.org/2022.semeval-1.74.pdf>`_     | `[CodaLab] <https://competitions.codalab.org/competitions/34175>`_                                             | 2022 | 10001 |               |
+------------------------------+-----------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+------+-------+---------------+

Converting Your Dataset
~~~~~~~~~~~~~~~~~~~~~~~
To prepare your local copy of the any of the above datasets for library usage, use the scripts provided under ``tools/conversion``. For example, use::

  python3 convert_harmemes.py --dataset-dir /path/to/dataset_dir --processed-dir /path/to/processed_dir

where ``dataset-dir`` is the directory containing your raw HarMeme dataset and ``processed-dir`` is the directory that should hold the converted MAMI dataset.

+------------------------------+----------------------------+
| Dataset                      | Script                     |
+==============================+============================+
| Facebook Hateful Memes (FHM) | convert_fhm.py             |
+------------------------------+----------------------------+
| Fine grained FHM             | convert_finegrained_fhm.py |
+------------------------------+----------------------------+
| HarMeme                      | convert_harmemes.py        |
+------------------------------+----------------------------+
| Harm-C                       | convert_harmc.py           |
+------------------------------+----------------------------+
| Harm-P                       | convert_harmp.py           |
+------------------------------+----------------------------+
| MAMI                         | convert_mami.py            |
+------------------------------+----------------------------+

For the MAMI dataset, we also provide the option to preprocess the dataset, ie; removing punctuation, removing non-ASCII characters, remove URLs, remove extra whitespaces, etc. To preprocess, set ``--process-data True``::

  python3 convert_mami.py --dataset-dir /path/to/dataset_dir --processed-dir /path/to/processed_dir --process-data True

Adding Custom Datasets
~~~~~~~~~~~~~~~~~~
Each custom meme dataset is required to have the following fields:

* img: image filepath
* text: superimposed/overlaid text
* {labels}: the label name changes based on the dataset (i.e. hateful, offensive)

Make sure your custom dataset folder's tree looks similar to the following:

.. code-block:: text

   processed_dir
   ├── annotations
   │   ├── test.jsonl
   │   ├── train.jsonl
   │   └── validate.jsonl
   └── images
      ├── 1.jpg
      └── 2.jpg


**************************
Meme Models and Evaluation
**************************
Supported Language Models
~~~~~~~~~~~~~~~~~~~~~~~~~~
+------------+-------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+-------+
| Model      | Paper                                                       | Source                                                                                                               | Year  |
+============+=============================================================+======================================================================================================================+=======+
| BART       | `[arxiv] <https://aclanthology.org/2020.acl-main.703.pdf>`_ | `[HuggingFace] <https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForConditionalGeneration>`_ | 2019  |
+------------+-------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+-------+
| PromptHate | `[arxiv] <https://arxiv.org/pdf/2302.04156.pdf>`_           | `[GitLab] <https://gitlab.com/bottle_shop/safe/prompthate>`_                                                         | 2022  |
+------------+-------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+-------+

Supported Vision-Language Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
+------------+---------------------------------------------------+----------------------------------------------------------------------------------------------------------------+------+
| Model      | Paper                                             | Source                                                                                                         | Year |
+============+===================================================+================================================================================================================+======+
| VisualBERT | `[arxiv] <https://arxiv.org/pdf/1908.03557.pdf>`_ | `[HuggingFace] <https://huggingface.co/docs/transformers/model_doc/visual_bert#transformers.VisualBertModel>`_ | 2019 |
+------------+---------------------------------------------------+----------------------------------------------------------------------------------------------------------------+------+
| LXMERT     | `[arxiv] <https://arxiv.org/pdf/1908.07490.pdf>`_ | `[HuggingFace] <https://huggingface.co/docs/transformers/model_doc/lxmert#transformers.LxmertModel>`_          | 2019 |
+------------+---------------------------------------------------+----------------------------------------------------------------------------------------------------------------+------+
| VL-T5      | `[arxiv] <https://arxiv.org/pdf/2102.02779.pdf>`_ | `[GitHub] <https://github.com/j-min/VL-T5>`_                                                                   | 2021 |
+------------+---------------------------------------------------+----------------------------------------------------------------------------------------------------------------+------+
| FLAVA      | `[arxiv] <https://arxiv.org/pdf/2112.04482.pdf>`_ | `[HuggingFace] <https://huggingface.co/docs/transformers/model_doc/flava#transformers.FlavaModel>`_            | 2021 |
+------------+---------------------------------------------------+----------------------------------------------------------------------------------------------------------------+------+

Model configuration
~~~~~~~~~~~~~~~~~~~
Once your dataset is converted, follow these steps to configure the desired model for usage:
#. Go to ``configs`` and pick the relevant dataset folder.
#. Choose the YAML file relevant to the desired model.
#. Look for the ``annotation_filepaths`` key and modify the values for ``train``,``test``,``predict``,``validation`` based on your ``processed_dir``.
#. (Optional) If you wish to modify any of the training hyperparameters, look for the ``trainer`` key and modify the values as required.


Model Usage
~~~~~~~~~~~
Once you have created your model configuration, follow these steps to use the configured model:
#. Go to ``scripts`` and pick the relevant dataset folder.
#. Pick ``test`` or ``train`` based on your requirement and locate the script for your model.

For example, if you wish to train FLAVA on FHM dataset, run the following command::

  bash scripts/fhm/train/flava.sh


MATK Overview
~~~~~~~~~~~~~~
+------------------+---------------+---------------+---------------+---------------+----------------------------------------------------+
|                  | BART          | FLAVA         | LXMERT        | VisualBERT    | Remarks                                            |
+==================+===============+===============+===============+===============+====================================================+
| FHM              | |green_check| | |green_check| | |green_check| | |green_check| |                                                    |
+------------------+---------------+---------------+---------------+---------------+----------------------------------------------------+
| Fine Grained FHM | |green_check| | |green_check| | |green_check| | |green_check| | Protected target and protected group not supported |
+------------------+---------------+---------------+---------------+---------------+----------------------------------------------------+
| MAMI             | |green_check| | |green_check| | |green_check| | |green_check| |                                                    |
+------------------+---------------+---------------+---------------+---------------+----------------------------------------------------+
| HarMeme          | |green_check| | |green_check| | |green_check| | |green_check| |                                                    |
+------------------+---------------+---------------+---------------+---------------+----------------------------------------------------+
| Harm-C + Harm-P  | |green_check| | |green_check| | |green_check| | |green_check| |                                                    |
+------------------+---------------+---------------+---------------+---------------+----------------------------------------------------+

Model Performance
~~~~~~~~~~~~~~
Coming soon...

**************************
Meme Models Analysis
**************************


**************************
Authors and acknowledgment
**************************

*  Ming Shan HEE, Singapore University of Technology and Design (SUTD)
*  Aditi KUMARESAN, Singapore University of Technology and Design (SUTD)
*  Nirmalendu PRAKASH, Singapore University of Technology and Design (SUTD)
*  Rui CAO, Singapore Management University (SMU)
*  Prof. Roy Ka-Wei LEE, Singapore University of Technology and Design (SUTD)

**************************
License
**************************

Coming soon...

**************************
Project status
**************************
[] Dataset Preprocessing
[] README.rst updates
[] Implementing analysis code