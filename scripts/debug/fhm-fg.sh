# #!/bin/bash

# python3 main.py \
#     +experiment=fhm_finegrained/flava.yaml \
#     action=fit \
#     trainer=debug_trainer

# python3 main.py \
#     +experiment=fhm_finegrained/visualbert.yaml \
#     action=fit \
#     trainer=debug_trainer

# python3 main.py \
#     +experiment=fhm_finegrained/lxmert.yaml \
#     action=fit \
#     trainer=debug_trainer

python3 main.py \
    +experiment=fhm_finegrained/t5_classification.yaml \
    action=fit \
    trainer=debug_trainer

python3 main.py \
    +experiment=fhm_finegrained/bart_classification.yaml \
    action=fit \
    trainer=debug_trainer

