#!/bin/bash

python3 main.py \
    +experiment=fhm/visualbert.yaml \
    action=fit \
    trainer=debug_trainer

python3 main.py \
    +experiment=fhm/visualbert.yaml \
    action=fit \
    trainer=debug_trainer

python3 main.py \
    +experiment=fhm/lxmert.yaml \
    action=fit \
    trainer=debug_trainer


python3 main.py \
    +experiment=fhm/t5_classification.yaml \
    action=fit \
    trainer=debug_trainer

python3 main.py \
    +experiment=fhm/bart_classification.yaml \
    action=fit \
    trainer=debug_trainer

