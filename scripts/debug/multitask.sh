python3 main.py --multirun \
    +experiment=multitask/flava.yaml \
    action=fit \
    trainer=debug_trainer

python3 main.py --multirun \
    +experiment=multitask/visualbert.yaml \
    action=fit \
    trainer=debug_trainer

python3 main.py --multirun \
    +experiment=multitask/t5_classification.yaml \
    action=fit \
    trainer=debug_trainer \
    datamodule.batch_size=4 \