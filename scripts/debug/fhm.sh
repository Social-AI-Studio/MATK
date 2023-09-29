# # Fine-tunes Flava on Fine-Grain FHM
# python3 main.py --multirun \
#     +experiment=fhm_finegrained_flava.yaml \
#     action=fit \
#     seed_everything=1111,1112,1113,1114,1115,1116,1117,1118,1119,1120

# python3 main.py --multirun \
#     +experiment=fhm_finegrained_flava.yaml \
#     action=fit \
#     seed_everything=1111,1112,1113,1114,1115,1116,1117,1118,1119,1120


python3 main.py --multirun \
    +experiment=fhm/visualbert.yaml \
    action=fit \
    trainer=debug_trainer

python3 main.py --multirun \
    +experiment=fhm/lxmert.yaml \
    action=fit \
    trainer=debug_trainer