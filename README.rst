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
3. Create a new YAML config file and script that will reference your new dataset class and paths to your dataset files.

**************************
Meme Models and Evaluation
**************************
Supported Language Models
~~~~~~~~~~~~~~~~~~~~~~~~~~
+------------+-------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+-------+-----------------+
| Model      | Paper                                                       | Source                                                                                                               | Year  | Name in toolkit |
+============+=============================================================+======================================================================================================================+=======+=================+
| BART       | `[arxiv] <https://aclanthology.org/2020.acl-main.703.pdf>`_ | `[HuggingFace] <https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForConditionalGeneration>`_ | 2019  | bart            |
+------------+-------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+-------+-----------------+
| PromptHate | `[arxiv] <https://arxiv.org/pdf/2302.04156.pdf>`_           | `[GitLab] <https://gitlab.com/bottle_shop/safe/prompthate>`_                                                         | 2022  | t5              |
+------------+-------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+-------+-----------------+

Supported Vision-Language Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
+------------+---------------------------------------------------+----------------------------------------------------------------------------------------------------------------+------+-----------------+
| Model      | Paper                                             | Source                                                                                                         | Year | Name in toolkit |
+============+===================================================+================================================================================================================+======+=================+
| VisualBERT | `[arxiv] <https://arxiv.org/pdf/1908.03557.pdf>`_ | `[HuggingFace] <https://huggingface.co/docs/transformers/model_doc/visual_bert#transformers.VisualBertModel>`_ | 2019 | visualbert      |
+------------+---------------------------------------------------+----------------------------------------------------------------------------------------------------------------+------+-----------------+
| LXMERT     | `[arxiv] <https://arxiv.org/pdf/1908.07490.pdf>`_ | `[HuggingFace] <https://huggingface.co/docs/transformers/model_doc/lxmert#transformers.LxmertModel>`_          | 2019 | lxmert          |
+------------+---------------------------------------------------+----------------------------------------------------------------------------------------------------------------+------+-----------------+
| VL-T5      | `[arxiv] <https://arxiv.org/pdf/2102.02779.pdf>`_ | `[GitHub] <https://github.com/j-min/VL-T5>`_                                                                   | 2021 | vlt5            |
+------------+---------------------------------------------------+----------------------------------------------------------------------------------------------------------------+------+-----------------+
| FLAVA      | `[arxiv] <https://arxiv.org/pdf/2112.04482.pdf>`_ | `[HuggingFace] <https://huggingface.co/docs/transformers/model_doc/flava#transformers.FlavaModel>`_            | 2021 | flava           |
+------------+---------------------------------------------------+----------------------------------------------------------------------------------------------------------------+------+-----------------+

Usage
~~~~~
+----------------------+-------------------+--------------------+----------------------+
| Datamodule           | Dataset           | Usage              |  Name in toolkit     |
+======================+===================+====================+======================+
| FasterRCNNDataModule | FasterRCNNDataset | LXMERT, VisualBERT | FasterRCNNDataModule |
+----------------------+-------------------+--------------------+----------------------+
| ImagesDataModule     | ImagesDataset     | FLAVA              | ImagesDataModule     |
+----------------------+-------------------+--------------------+----------------------+
| TextDataModule       | TextDataset       | BART, T5           | TextDataModule       |
+----------------------+-------------------+--------------------+----------------------+

Make sure you have already run::

   pip install -r requirements.txt

To train the VisualBERT model on HarMeme (intensity subtask)::

   python main.py --model visualbert --dataset harmeme --task intensity --datamodule FasterRCNNDataModule --action fit

To test the VisualBERT model on the Facebook Hateful Memes Dataset::

   python main.py --model visualbert --dataset harmeme --task intensity --datamodule FasterRCNNDataModule --action test


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
Facebook Hateful Memes Dataset

+------------+---------------+-----------------+----------------+------------------+
| Model      | label_val_acc | label_val_auroc | label_test_acc | label_test_auroc |
+============+===============+=================+================+==================+
| FLAVA      | 0.658         | 0.768           | 0.702          | 0.787            |
+------------+---------------+-----------------+----------------+------------------+
| LXMERT     | 0.604         | 0.669           | 0.642          | 0.7007           |
+------------+---------------+-----------------+----------------+------------------+
| Visualbert | 0.646         | 0.686           | 0.6439         | 0.6895           |
+------------+---------------+-----------------+----------------+------------------+

Finegrained FHM

+------------+--------------+----------------+---------------+-----------------+
| Model      | hate_val_acc | hate_val_auroc | hate_test_acc | hate_test_auroc |
+============+==============+================+===============+=================+
| FLAVA      | 0.692        | 0.76           | 0.6759        | 0.7882          |
+------------+--------------+----------------+---------------+-----------------+
| LXMERT     | 0.612        | 0.676          | 0.634         | 0.6862          |
+------------+--------------+----------------+---------------+-----------------+
| VisualBERT | 0.606        | 0.693          | 0.6399        | 0.6971          |
+------------+--------------+----------------+---------------+-----------------+

HarMeme - intensity subtask

+------------+-------------------+---------------------+--------------------+----------------------+
| Model      | intensity_val_acc | intensity_val_auroc | intensity_test_acc | intensity_test_auroc |
+============+===================+=====================+====================+======================+
| FLAVA      | 0.627             | 0.728               | 0.8135             | 0.879                |
+------------+-------------------+---------------------+--------------------+----------------------+
| LXMERT     | 0.661             | 0.711               | 0.7598             | 0.7782               |
+------------+-------------------+---------------------+--------------------+----------------------+
| VisualBert | 0.661             | 0.696               | 0.7598             | 0.8149               |
+------------+-------------------+---------------------+--------------------+----------------------+

HarMeme - target subtask

+------------+----------------+------------------+-----------------+-------------------+
| Model      | target_val_acc | target_val_auroc | target_test_acc | target_test_auroc |
+============+================+==================+=================+===================+
| FLAVA      | 0.59           | 0.517            | 0.9516          | 0.7439            |
+------------+----------------+------------------+-----------------+-------------------+
| LXMERT     | 0.607          | 0.529            | 0.8467          | 0.6708            |
+------------+----------------+------------------+-----------------+-------------------+
| VisualBert | 0.705          | 0.551            | 0.9596          | 0.7025            |
+------------+----------------+------------------+-----------------+-------------------+

MAMI - subtask_a (misogyny detection)
+------------+--------------------+----------------------+---------------------+-----------------------+
| Model      | misogynous_val_acc | misogynous_val_auroc | misogynous_test_acc | misogynous_test_auroc |
+============+====================+======================+=====================+=======================+
| FLAVA      | 0.9                | 0.948                | 0.7229              | 0.8202                |
+------------+--------------------+----------------------+---------------------+-----------------------+
| LXMERT     | 0.89               | 0.945                | 0.683               | 0.7552                |
+------------+--------------------+----------------------+---------------------+-----------------------+
| VisualBERT | 0.92               | 0.948                | 0.7039              | 0.7711                |
+------------+--------------------+----------------------+---------------------+-----------------------+

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
