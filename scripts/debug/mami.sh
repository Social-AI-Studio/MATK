# #!/bin/bash

python3 main.py \
    +experiment=mami/flava.yaml \
    action=fit \
    trainer=debug_trainer

python3 main.py \
    +experiment=mami/visualbert.yaml \
    action=fit \
    trainer=debug_trainer

python3 main.py \
    +experiment=mami/lxmert.yaml \
    action=fit \
    trainer=debug_trainer

python3 main.py \
    +experiment=mami/t5_classification.yaml \
    action=fit \
    trainer=debug_trainer 

python3 main.py \
    +experiment=mami/bart_classification.yaml \
    action=fit \
    trainer=debug_trainer

