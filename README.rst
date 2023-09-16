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

* Provides a framework for training and evaluating a different language and vision-language models on well known hateful memes datasets.
* Allows for efficient experimentation and parameter tuning through modification of configuration files. 
* Evaluate models using different state-of-the-art evaluation metrics such as Accuracy and AUROC. 
* Supports visualization by integrating with Tensorboard, allowing users to easily view and analyze metrics in a user-friendly GUI.


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

Adding Custom Datasets
~~~~~~~~~~~~~~~~~~~~~~
1. To use a dataset lot listed above, copy the code given in one of the dataset files, eg; ``datamodules/datasets/fhm.py``. 
2. Modify the base class implementation, specifically ``_preprocess_annotations`` to suit your dataset's needs.
3. Follow the steps in model usage to use the custom dataset in your experiment.

************************************
Supported Meme Models and Evaluation
************************************

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


***********
Model Usage
***********

To configure the different elements of the toolkit, we use ``Hydra``, an open-source Python framework that simplifies the development of research and other complex applications. 
Its key feature is the ability to dynamically create a hierarchical configuration by composition and override it through both config files and the command line.

Step 1: Configure Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~

The purpose of the dataset class is to store the samples and their corresponding labels. Within this dataset class we:
- preprocess the annotations: remove any hyperlinks, standardize label names, remove samples without labels, etc
- load any auxiliary information: for example, .pkl files of captions for each image
- load features

For each dataset, we support the following dataset types ``FRCNNDataset``, ``ImageDataset``, and ``TextClassificationDataset``. 

+---------------------------+------------------------+-----------------------------------------------------------------------------------------------------------------------+
| Dataset                   | Usage                  | Remarks                                                                                                               |
+===========================+========================+=======================================================================================================================+
| FasterRCNNDataset         | For LXMERT, VisualBert | To handle `Faster-RCNN <https://github.com/eladsegal/gqa_lxmert/blob/main/notebook.ipynb>`_ features of images + text |
+---------------------------+------------------------+-----------------------------------------------------------------------------------------------------------------------+
| ImageDataset              | For FLAVA, VisualBert  | To handle raw images + text                                                                                           |
+---------------------------+------------------------+-----------------------------------------------------------------------------------------------------------------------+
| TextClassificationDataset | For T5                 | To handle text                                                                                                        |
+---------------------------+------------------------+-----------------------------------------------------------------------------------------------------------------------+



To configure the dataset, go to ``configs/dataset``, pick the file based on your dataset choice and specify:

- ``annotation_filepaths (dict)``
- ``image_dirs (dict)``
- ``auxiliary_dicts (dict)``
- ``feats_dir (dict)``

For all other optional parametesr listed below please refer to the experiment config files we provide:

- ``dataset_class``: class path of the dataset you choose, eg; ``datasets.fhm.ImageDataset``.
- ``text_template``
- ``labels (list)``

Step 2: Configure DataModule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The data modules initialize the tokenizer and the data loaders (which handle batch size, number of workers, etc.).

To configure the datamodule, go to ``configs/datamodule`` and pick the file based on your model choice and specify:

- ``shuffle_train (bool)``: set to True to make sure we aren’t exposing our model to the same cycle (order) of data in every epoch
- ``num_workers (int)``: how many subprocesses to use for data loading
- ``batch_size (int)``: the number of samples the model processes at once during training
- ``class_path``: class path of the datamodule you choose (e.g., ``datamodules.frcnn_datamodule.FRCNNDataModule``).

The following parameters are specified as '???' because they are specific to the experiment configuration:

- ``tokenizer_class_or_path``: class or path of the pretrained tokenizer (e.g., ``t5-large``).

Step 3: Configure Model
~~~~~~~~~~~~~~~~~~~~~~~

To configure an existing model, go to ``configs/model`` and pick the file based on your model choice. The following parameters need to be specified:

- ``class_path``: class path of the model you chose (e.g., ``models.flava.FlavaClassificationModel``).
- ``model_class_or_path``: class or path of the pretrained model (e.g., ``facebook/flava-full``).

For all other optional parametesr listed below please refer to the experiment config files we provide:

- ``cls_dict (dict)``: dictionary where each key-value pair is defined as ``{label}:#number of class``.
- ``optimizers``

Step 4: Configure Trainer
~~~~~~~~~~~~~~~~~~~~~~~~~

The Trainer helps automate several aspects of training. It handles all loop details for you, including:

- Automatically enabling/disabling gradients.
- Running the training, validation, and test data loaders.
- Calling the Callbacks at the appropriate times.
- Putting batches and computations on the correct devices.

To configure the trainer, go to ``configs/trainer``, pick the trainer of your choice. Below are the **required** parameters and the **default** values we use. 
You can also tweak the trainer by adding parameters from here: `[Trainer API] <https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api>`_

- ``accelerator``: ``cuda``
- ``max_epochs (int)``: ``30``
- ``enable_checkpointing (bool)``: ``True``
- ``logger``
- ``callbacks``

Step 5: Configure Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To configure your experiment, you can take a look at any of the dataset folders under ``configs/experiment``. The following parameters need to be specified:

- ``defaults``: This is a list in our input config that instructs Hydra on how to build the output config. The Defaults List is ordered:

  - If multiple configs define the same value, the last one wins.
  - If multiple configs contribute to the same dictionary, the result is the combined dictionary.

The following parameters contribute to the parameter dictionaries of the values defined in the defaults list. Remember, some of these had keys that have '???' as their values. Taking the example of FLAVA on FHM:

- ``cls_dict (dict)``: dictionary where each key-value pair is defined as ``{label}:#number of class``. For FHM, the label is called 'label,' and it can take 2 values.
- ``optimizers``
- ``dataset_class``: class path of the dataset class you're using, eg;  ``datasets.fhm..
- ``text_template``
- ``labels (list)``: the list of labels in the dataset; in this case, 'label' is the only label.
- ``processor_class_path``: class path of the pretrained image processor.
- ``monitor_metric``: metric to monitor. Metrics are generated as `{stage}_{label_name}_{type}`
- ``monitor_mode``: one of {min, max} - the decision to overwrite the saved file is made based on the maximization/minimization of the monitored metric
- ``save_top_ks (int)``:  the best k models to save according to the metric monitored
- ``experiment_name``

Job Settings

- ``hydra.verbose``
- ``seed_everything (int)``
- ``overwrite``
- ``action``: Specifies whether you are training or testing a model. Can be specified at runtime.

Step 6: Running your Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To test your configurations for correctness, you can use ``debug trainer``:

.. code-block:: bash
  python3 main.py --multirun \
    +experiment={experiment config location} \
    action=fit \
    trainer=debug_trainer

To run **training**, you can use ``single_gpu_trainer`` or ``multi_gpu_trainer``:

.. code-block:: bash

  python3 main.py --multirun \
    +experiment={experiment config location} \
    action=fit \
    trainer={single_gpu_trainer, multi_gpu_trainer}

For example, to **train** VisualBERT on FHM using the ``multi_gpu_trainer``:

.. code-block:: bash

  python3 main.py --multirun \
    +experiment=fhm/visualbert.yaml \
    action=fit \
    trainer=multi_gpu_trainer

Similarly, you can run **inference** by changing ``action`` to ``test``:

.. code-block:: bash

  python3 main.py --multirun \
    +experiment={experiment config location} \
    action=test \
    trainer={single_gpu_trainer, multi_gpu_trainer}

For example, to run **inference** for VisualBERT on FHM:

.. code-block:: bash

  python3 main.py --multirun \
    +experiment={experiment config location} \
    action=test \
    trainer={single_gpu_trainer, multi_gpu_trainer}

*****************
Model Performance
*****************
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
