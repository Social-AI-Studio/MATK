# python3 main.py --multirun \
#     +experiment=multitask/flava.yaml \
#     action=fit \
#     trainer=debug_trainer

# python3 main.py --multirun \
#     +experiment=multitask/visualbert.yaml \
#     action=fit \
#     trainer=debug_trainer

python3 main.py --multirun \
    +experiment=multitask/t5_clm.yaml \
    action=fit \
    trainer=debug_trainer \