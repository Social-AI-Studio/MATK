MATK: Meme Analysis ToolKit
===========================

MATK (Meme Analysis Toolkit) aims at training, analyzing and comparing
the state-of-the-art Vision Language Models on the various downstream
memes tasks (i.e. hateful memes classification, attacked group
classification, hateful memes explanation generation).

.. contents:: Table of Contents 
   :depth: 2


***************
Main Features
***************

* Provides a framework for training and evaluating a different language and vision-language models on well known hateful memes datasets.
* Allows for efficient experimentation and parameter tuning through modification of configuration files. 
* Evaluates models using different state-of-the-art evaluation metrics such as Accuracy and AUROC. 
* Supports visualization by integrating with Tensorboard, allowing users to easily view and analyze metrics in a user-friendly GUI.


**************
MATK Overview
**************
.. |green_check| unicode:: U+2714
   :trim:

+-------------------------------------------------------------------------------------------------------+----------------------------------------------------------+-------------------------------------------------+--------------------------------------------------+------------------------------------------------------+----------------------------------------------------+
|                                                                                                       | `BART <https://aclanthology.org/2020.acl-main.703.pdf>`_ | `FLAVA <https://arxiv.org/pdf/2112.04482.pdf>`_ | `LXMERT <https://arxiv.org/pdf/1908.07490.pdf>`_ | `VisualBERT <https://arxiv.org/pdf/1908.03557.pdf>`_ | Remarks                                            |
+=======================================================================================================+==========================================================+=================================================+==================================================+======================================================+====================================================+
| `FHM <https://www.drivendata.org/accounts/login/?next=/competitions/70/hateful-memes-phase-2/data/>`_ | |green_check|                                            | |green_check|                                   | |green_check|                                    | |green_check|                                        |                                                    |
+-------------------------------------------------------------------------------------------------------+----------------------------------------------------------+-------------------------------------------------+--------------------------------------------------+------------------------------------------------------+----------------------------------------------------+
| `Fine grained FHM <https://github.com/facebookresearch/fine_grained_hateful_memes/tree/main/data>`_   | |green_check|                                            | |green_check|                                   | |green_check|                                    | |green_check|                                        | Protected target and protected group not supported |
+-------------------------------------------------------------------------------------------------------+----------------------------------------------------------+-------------------------------------------------+--------------------------------------------------+------------------------------------------------------+----------------------------------------------------+
| `HarMeme <https://github.com/di-dimitrov/harmeme>`_                                                   | |green_check|                                            | |green_check|                                   | |green_check|                                    | |green_check|                                        |                                                    |
+-------------------------------------------------------------------------------------------------------+----------------------------------------------------------+-------------------------------------------------+--------------------------------------------------+------------------------------------------------------+----------------------------------------------------+
| `Harm-C + Harm-P <https://github.com/LCS2-IIITD/MOMENTA>`_                                            | |green_check|                                            | |green_check|                                   | |green_check|                                    | |green_check|                                        |                                                    |
+-------------------------------------------------------------------------------------------------------+----------------------------------------------------------+-------------------------------------------------+--------------------------------------------------+------------------------------------------------------+----------------------------------------------------+
| `MAMI <https://competitions.codalab.org/competitions/34175>`_                                         | |green_check|                                            | |green_check|                                   | |green_check|                                    | |green_check|                                        |                                                    |
+-------------------------------------------------------------------------------------------------------+----------------------------------------------------------+-------------------------------------------------+--------------------------------------------------+------------------------------------------------------+----------------------------------------------------+


************
Installation
************

To get started, run the following command::

  pip install -r requirements.txt


For installation instructions related to image feature extraction, inpainting and captioning, please refer to the ``preprocessing`` directory. 


***************
Getting Started
***************

This section will cover how to use the toolkit to execute model training and inference using the currently supported models and datasets. 
The toolkit uses `Hydra <https://hydra.cc/docs/intro/>`_ framework, ensuring a composable and hierarchical configuration setup. 
The preconfigured settings for the existing models and datasets are available within the `configs/experiments` directory.

Dataset Set-Up
--------------

Although preconfigured settings for both models and datasets have been provided, you will need to (1) `download the datasets <#matk-overview>`_ 
and (2) update the directory paths pertaining to the datasets and their accompanying auxiliary information.
Once you have downloaded the dataset, identify and update the respective configuration file under `dataset` folder (i.e., fhm_finegrained)


Model Training
--------------

Subsequently, once you have identified the respective configuraton file (i.e., fhm_finegrained_flava), you can train the model using the following commands:

.. code-block:: bash

  python3 main.py \
    +experiment=fhm_finegrained_flava \
    action=train


Model Inference
---------------

Similarly, you can run the model on your test set using the following command:

.. code-block:: bash

  python3 main.py \
    +experiment=fhm_finegrained_flava \
    action=test


Simple Configuration Overriding
-------------------------------

If you encounter issues stemming from hardware limitations or want to experiment with alternative hyperparameters, 
you have the option to modify the settings through either (1) the composed configuration file or (2) the command line interface in the terminal. 
For executing one-time override commands, utilize the following command:

.. code-block:: bash

  python3 main.py \
    +experiment=fhm_finegrained_flava \
    action=test \
    datamodule.batch_size=16 \
    trainer.accumulate_grad_batches=1 \
    model.optimizers.0.lr=2e-5


*************************************************
Advanced Usage (Implementing New Dataset / Model)
*************************************************

As researchers, you may wish to introduce and experiment with either new models or new datasets. 
MATK offers an intuitive and modular framework equipped with designated components to streamline such implementations.

Framework Outline
-----------------

The illustration outlines the core configurations and python code used in the composed `experiments` configuration.

::

    MATK
    ├──configs
    ├──── dataset
    ├──── datamodule
    ├──── model
    ├──── metric      
    └──── trainer
    ├── datasets
    ├── datamodules
    └── models


Implementing New Dataset
------------------------

To introduce a new dataset (i.e., fhm_finegrained), it is necessary to generate the following files:

- `dataset/fhm_finegrained.py` 
- `configs/dataset/fhm_finegrained.yaml`


Python Code
~~~~~~~~~~~

The Python code facilitates (1) the loading of annotation files, (2) the loading of auxiliary files, and (3) performing dataset preprocessing (i.e., stopwords removal, lowercase). 
To establish a unified interface for diverse model types, including unimodal and multimodal models, three common base classes are introduced in datasets/base.py: "ImageBase," "FeatureBase," and "TextBase."

For most use cases, you can inherit one of these three base classes and implement the required core functions:

- __len__(self)
- __getitem__(self, idx: int)

You can examine the existing implementations under the `dataset` folder for reference.

YAML Configuration
~~~~~~~~~~~~~~~~~~

The configuration file stores the filepaths to the dataset and the relevant auxiliary information. In essense, you are required to provide:

- ``annotation_filepaths (dict)``
- ``image_dirs (dict)``
- ``auxiliary_dicts (dict)``
- ``feats_dir (dict)``


Implementing New Model
-----------------------

To introduce a new model (i.e., flava), it is necessary to generate the following files:

- models/flava.py
- configs/model/flava.yaml


Python Code
~~~~~~~~~~~

The Python code controls (1) the model architecture and (2) the various model training stages (i.e., train, validation and test). Under the hood, we used Pytorch's LightningModule to handle these processes. 

You can examine the existing implementations under the `models` folder for reference.

YAML Configuration
~~~~~~~~~~~~~~~~~~

The configuration file defines the model classes and handles the the models' hyperparameters.


*****************
Model Performance
*****************

+------------+---------------+-----------------+---------------+---------------+
| AUROC      | FHM           | FHM Finegrained | HarMeme       | MAMI          |
+============+===============+=================+===============+===============+
| LXMERT     | 0.689 (0.014) | 0.680 (0.007)   | 0.818 (0.014) | 0.763 (0.007) |
+------------+---------------+-----------------+---------------+---------------+
| VisualBERT | 0.708 (0.014) | 0.672 (0.013)   | 0.821 (0.015) | 0.779 (0.007) |
+------------+---------------+-----------------+---------------+---------------+
| FLAVA      | 0.786 (0.009) | 0.765 (0.011)   | 0.846 (0.015) | 0.803 (0.006) |
+------------+---------------+-----------------+---------------+---------------+


The AUROC scores are presented in the format `average (std.dev)`, where both the average and standard deviation values are calculated across 10 random seeds, ranging from 1111 to 1120.

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
