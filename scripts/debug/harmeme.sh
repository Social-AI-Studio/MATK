# #!/bin/bash

python3 main.py \
    +experiment=harmeme/flava.yaml \
    action=fit \
    trainer=debug_trainer

python3 main.py \
    +experiment=harmeme/visualbert.yaml \
    action=fit \
    trainer=debug_trainer

python3 main.py \
    +experiment=harmeme/lxmert.yaml \
    action=fit \
    trainer=debug_trainer

python3 main.py \
    +experiment=harmeme/t5_classification.yaml \
    action=fit \
    trainer=debug_trainer 

python3 main.py \
    +experiment=harmeme/bart_classification.yaml \
    action=fit \
    trainer=debug_trainer

