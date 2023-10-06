#!/bin/bash
# python3 main.py --multirun \
#     +experiment=fhm_finegrained/bart_classification.yaml \
#     action=fit \
#     trainer=debug_trainer

# python3 main.py --multirun \
#     +experiment=harmeme/bart_classification.yaml \
#     action=fit \
#     trainer=debug_trainer

# python3 main.py --multirun \
#     +experiment=mami/bart_classification.yaml \
#     action=fit \
#     trainer=debug_trainer

# python3 main.py --multirun \
#     +experiment=fhm_finegrained/t5_classification.yaml \
#     action=fit \
#     trainer=debug_trainer

# python3 main.py --multirun \
#     +experiment=harmeme/t5_classification.yaml \
#     action=fit \
#     trainer=debug_trainer

python3 main.py --multirun \
    +experiment=mami/t5_classification.yaml \
    action=fit \
    trainer=debug_trainer