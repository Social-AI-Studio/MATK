MATK: Meme Analysis ToolKit
===========================

MATK (Meme Analysis Toolkit) aims at training, analyzing and comparing
the state-of-the-art Vision Language Models on the various downstream
memes tasks (i.e. hateful memes classification, attacked group
classification, hateful memes explanation generation).

.. contents:: Table of Contents 
   :depth: 2

************
Installation
************

To get started, run the following command::

  pip install -r requirements.txt


For installation instructions related to image feature extraction, inpainting and captioning, please refer to the ``preprocessing`` directory. 

***************
Main Features
***************

* Provides a framework for training and evaluating a different language and vision-language models on well known hateful memes datasets.
* Allows for efficient experimentation and parameter tuning through modification of configuration files. 
* Evaluates models using different state-of-the-art evaluation metrics such as Accuracy and AUROC. 
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


**************
Beginner Usage
**************

This section will cover how to use the toolkit to run training and inference with the currently supported models and datasets. 
For more advanced usage, such as evaluating a model on a custom dataset or a introducing a new model, please go to the Advanced Usage section.

To configure the different elements of the toolkit, we use ``Hydra``, an open-source Python framework that simplifies the development of complex research applications. 
Its key feature is the ability to dynamically create a hierarchical configuration by composition and override it through both config files and the command line.

Step 1: Configure Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~

The purpose of the dataset class is to store the samples and their corresponding labels. Within this dataset class we:

- preprocess the annotations: remove any hyperlinks, standardize label names, remove samples without labels, etc
- load any auxiliary information: for example, .pkl files of captions for each image
- load features

For each dataset, we support the following dataset types: ``FRCNNDataset``, ``ImageDataset``, and ``TextClassificationDataset``. 

+---------------------------+------------------------+-----------------------------------------------------------------------------------------------------------------------+
| Dataset                   | Usage                  | Remarks                                                                                                               |
+===========================+========================+=======================================================================================================================+
| FasterRCNNDataset         | For LXMERT, VisualBert | To handle `Faster-RCNN <https://github.com/eladsegal/gqa_lxmert/blob/main/notebook.ipynb>`_ features of images + text |
+---------------------------+------------------------+-----------------------------------------------------------------------------------------------------------------------+
| ImageDataset              | For FLAVA, VisualBert  | To handle raw images + text                                                                                           |
+---------------------------+------------------------+-----------------------------------------------------------------------------------------------------------------------+
| TextClassificationDataset | For T5                 | To handle text                                                                                                        |
+---------------------------+------------------------+-----------------------------------------------------------------------------------------------------------------------+


To configure the dataset, go to ``configs/dataset`` and specify the following parameters in the dataset file:

- ``annotation_filepaths (dict)``
- ``image_dirs (dict)``
- ``auxiliary_dicts (dict)``
- ``feats_dir (dict)``

For all other optional parameters listed below please refer to the experiment config files in ``configs/experiment``:

- ``dataset_class``: class path of the dataset you choose, eg; ``datasets.fhm.ImageDataset``.
- ``text_template``
- ``labels (list)``

Step 2: Configure DataModule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The datamodules initialize the tokenizer and the data loaders (which handle batch size, number of workers, etc.).

To configure the datamodule, go to ``configs/datamodule`` and specify the following parameters in the datamodule file:

- ``shuffle_train (bool)``: set to True to make sure we aren’t exposing our model to the same cycle (order) of data in every epoch
- ``num_workers (int)``: how many subprocesses to use for data loading
- ``batch_size (int)``: the number of samples the model processes at once during training
- ``class_path``: class path of the datamodule you choose (e.g., ``datamodules.frcnn_datamodule.FRCNNDataModule``).

For all other optional parameters listed below please refer to the experiment config files in ``configs/experiment``:

- ``tokenizer_class_or_path``: class or path of the pretrained tokenizer (e.g., ``t5-large``).

Step 3: Configure Model
~~~~~~~~~~~~~~~~~~~~~~~

To configure a model, go to ``configs/model`` and specify the following parameters in the model file:

- ``class_path``: class path of the model you chose (e.g., ``models.flava.FlavaClassificationModel``).
- ``model_class_or_path``: class or path of the pretrained model (e.g., ``facebook/flava-full``).

For all other optional parameters listed below please refer to the experiment config files in ``configs/experiment``:

- ``cls_dict (dict)``: dictionary where each key-value pair is defined as ``{label}:{#number of class}``.
- ``optimizers``

Step 4: Configure Trainer
~~~~~~~~~~~~~~~~~~~~~~~~~

The Trainer helps automate several aspects of training. It handles all loop details for you, including:

- Automatically enabling/disabling gradients.
- Running the training, validation, and test data loaders.
- Calling the Callbacks at the appropriate times.
- Putting batches and computations on the correct devices.

To configure the trainer, go to ``configs/trainer``. Below are the **required** parameters and the **default** values we use. 
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

The following parameters specify values for parameters that were optional in their respective config files:

- ``cls_dict (dict)``
- ``optimizers``
- ``dataset_class``
- ``text_template``
- ``labels (list)``
- ``processor_class_path``: class path of the pretrained image processor, eg; ``facebook/flava-full``.
- ``monitor_metric``: metric to monitor. Metrics are generated as ``{stage}_{label_name}_{type}``
- ``monitor_mode``: one of ``{min, max}`` - the decision to overwrite the saved file is made based on the maximization/minimization of the monitored metric
- ``save_top_ks (int)``:  the best k models to save based on monitored metric .
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


**************
Advanced Usage
**************

This section will cover evaluating a model on a custom dataset and introducing a new model. 
For beginner usage, how to use the toolkit to run training and inference with the currently supported models and datasets, please go to the Beginner Usage section.

Add a new dataset
~~~~~~~~~~~~~~~~~

You will need to make the following changes in the ``datasets`` directory if you are a introducing a dataset named ABC.

#. Create a new file with the implementations of ``ABCBase`` and ``FRCNNDataset``, ``ImageDataset``, ``TextClassificationDataset``. Your ABCBase implementation should have the following structure:

    .. code-block:: python

        class ABCBase(Dataset):
            def __init__(
                self,
                annotation_filepath: str,
                auxiliary_dicts: dict,
                labels: List[str]
            ):

            def _preprocess_annotations(self, annotation_filepath: str):
                """
                Standardize label names, remove unlabelled samples, etc
                Args:
                    annotation_filepath (str): Path to the annotation file.

                Returns:
                    list: Processed annotations.
                """
            

            def _load_auxiliary(self, auxiliary_dicts: dict):
                """
                Load auxiliary data sources such as image captions

                Args:
                    auxiliary_dicts (dict): Dictionary of auxiliary data sources.

                Returns:
                    dict: Loaded auxiliary data.
                """
            
            def __len__(self):
                """
                Get the number of annotations in the dataset.

                Returns:
                    int: Number of annotations.
                """

    Next, the ``ImageDataset`` class must follow the following structure:

    .. code-block:: python

        class ImageDataset(ABCBase):
            def __init__(
                self,
                annotation_filepath: str,
                auxiliary_dicts: dict,
                labels: List[str],
                text_template: str,
                image_dir: str
            ):
                super().__init__(annotation_filepath, auxiliary_dicts, labels)

            def __getitem__(self, idx: int):
                """
                Get a specific item from the dataset.

                Args:
                    idx (int): Index of the item to retrieve.

                Returns:
                    dict: A dictionary containing data for the specified item.
                """
          

    Similarly, please mimic the implementations of ``FRCNNDataset`` and ``TextClassificationDataset``. You can follow ``datasets/fhm.py`` as an example.


#. Create a config file called abc.yaml inside ``configs/dataset`` for your dataset ABC. The key-value pairs in this file define the values each argument in your dataset class takes.
You can use ``configs/dataset/fhm.yaml`` as a reference. 

#. Here on, you can refer to this section: :ref:`Step 2: Configure DataModule`.

Add a new model
~~~~~~~~~~~~~~~

You will need to make the following changes in the ``models`` directory if you are a introducing a model named XYZ:

#. Your file should contain a model class with the following structure:

    ..code-block:: python

        class XYZClassificationModel(BaseLightningModule):
        def __init__(
            self,
            model_class_or_path: str,
            metrics_cfg: dict,
            cls_dict: dict,
            optimizers: list
        ):
            super().__init__()
            # set up classification
            # set up metric

        def training_step(self, batch, batch_idx):
            """
            Training step for the Flava classification model.

            Args:
                batch: Input batch from the data loader.
                batch_idx: Index of the current batch.

            Returns:
                torch.Tensor: Total loss for the batch.
            """
        
        def validation_step(self, batch, batch_idx):

        def test_step(self, batch, batch_idx): 

        def predict_step(self, batch, batch_idx):
        
        def configure_optimizers(self):
            """
            Configure optimizers for the Flava classification model.

            Returns:
                list: List of optimizer instances.
            """


#. Create a config file called xyz.yaml inside ``configs/model`` for your model XYZ. The key-value pairs in this file define the values each argument in your model class takes.
You can use ``configs/model/flava.yaml`` as a reference. 
        
#. Here on, you can refer to this section: :ref:`Step 2: Configure DataModule`.


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
