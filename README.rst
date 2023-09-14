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

Step 1: Configure Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~

For each dataset, we define a file with the classes - **FRCNNDataset**, **ImageDataset**, and **TextClassificationDataset**. The following table will help you choose the correct dataset class for your needs:

+-------------------+----------------------+----------------------------+
| Dataset           | DataModule           | Usage                      |
+===================+======================+============================+
| FasterRCNNDataset | FasterRCNNDataModule | For vision-language models |
+-------------------+----------------------+----------------------------+
| ImagesDataset     | ImagesDataModule     | For vision-language models |
+-------------------+----------------------+----------------------------+
| TextDataset       | TextDataModule       | For language models        |
+-------------------+----------------------+----------------------------+

Within this dataset class, we preprocess the annotations, load any auxiliary information, load features, and format the data for the task.

To configure the dataset, go to ``configs/dataset``, pick the file based on your dataset choice and specify:

- ``annotation_filepaths``: file paths containing the annotations for your dataset.
- ``image_dirs``:  directories containing the images for your dataset.
- ``auxiliary_dicts``: directories containing additional information like captions.
- ``feats_dir``: directories containing the features of your dataset's images.

The following parameters are specified as '???' because they are specific to the experiment configuration:

- ``dataset_class``: class path of ``FRCNNDataset``, ``ImageDataset``, and ``TextClassificationDataset``.
- ``text_template``: 
- ``labels``

Step 2: Configure DataModule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The data modules initialize the tokenizer and the data loaders (which handle batch size, number of workers, etc.).

To configure the datamodule, go to ``configs/datamodule`` and pick the file based on your model choice and specify:

- ``shuffle_train``: Based on your needs.
- ``num_workers``: Based on your needs.
- ``batch_size``: Based on your needs.
- ``class_path``: class path of the data module you choose.

The following parameters are specified as '???' because they are specific to the experiment configuration:

- ``tokenizer_class_or_path``

Step 3: Configure Model
~~~~~~~~~~~~~~~~~~~~~~~

To configure an existing model, go to ``configs/model`` and pick the file based on your model choice. The following parameters need to be specified:

- ``class_path``: Specifies the class path of the model you chose (e.g., ``models.flava.FlavaClassificationModel``).
- ``model_class_or_path``: Specifies the class or path of the pretrained model (e.g., ``facebook/flava-full``).

The following parameters are specified as '???' because they are specific to the experiment configuration:

- ``cls_dict``: Specifies a dictionary where each key-value pair is defined as `label : number of possible values`.
- ``optimizers``

Step 4: Configure Trainer
~~~~~~~~~~~~~~~~~~~~~~~~~

The Trainer helps automate several aspects of training. It handles all loop details for you, including:

- Automatically enabling/disabling gradients.
- Running the training, validation, and test data loaders.
- Calling the Callbacks at the appropriate times.
- Putting batches and computations on the correct devices.

To configure the trainer, go to ``configs/trainer``, pick the trainer of your choice and specify:

- ``accelerator``: device used for computations.
- ``max_epochs``
- ``enable_checkpointing``
- ``logger``
- ``callbacks``

Step 5: Configure Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To configure your experiment, you can take a look at any of the dataset folders under ``configs/experiment``. The following parameters need to be specified:

- ``defaults``: This is a list in our input config that instructs Hydra on how to build the output config. The Defaults List is ordered:

  - If multiple configs define the same value, the last one wins.
  - If multiple configs contribute to the same dictionary, the result is the combined dictionary.

The following parameters contribute to the parameter dictionaries of the values defined in the defaults list. Remember, some of these had keys that have '???' as their values. Taking the example of FLAVA on FHM:

- ``cls_dict``: Defines a dictionary of `{label_name}:{label_value}` pairs. For FHM, the label is called 'label,' and it can take 2 values.
- ``optimizers``: Specify based on requirements.
- ``dataset_class``: Class path of the dataset class you're using; in this case, ``ImageDataset`` from the ``fhm`` file under ``datasets``.
- ``text_template``
- ``labels``: Defines the list of labels in the dataset; in this case, 'label' is the only label.
- ``processor_class_path``: Class path of the pretrained image processor.
- ``monitor_metric``: Metrics are generated as `{stage}_{label_name}_{type}`. You can pick 1 metric to monitor.
- ``monitor_mode``: Specify based on requirements.
- ``save_top_ks``: Specify based on checkpoint requirements.
- ``experiment_name``: Name of the experiment you're running.

Job Settings

- ``hydra.verbose``
- ``seed_everything``
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


To run training, you can use ``single_gpu_trainer`` or ``multi_gpu_trainer``:

.. code-block:: bash

  python3 main.py --multirun \
    +experiment={experiment config location} \
    action=fit \
    trainer=single_gpu_trainer

Similarly, you can run inference by changing ``action`` to ``test``:

.. code-block:: bash

  python3 main.py --multirun \
    +experiment={experiment config location} \
    action=test \
    trainer=single_gpu_trainer



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
