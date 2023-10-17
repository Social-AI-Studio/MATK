#!/bin/bash
for model in mami
do
    python3 main.py --multirun \
        +experiment=fhm/flava.yaml \
        action=fit \
        datamodule.batch_size=16,32 \
        model.optimizers.0.lr=1e-4,3e-4,2e-5

    python3 main.py --multirun \
        +experiment=fhm/visualbert.yaml \
        action=fit \
        datamodule.batch_size=16,32 \
        model.optimizers.0.lr=1e-4,3e-4,2e-5

    python3 main.py --multirun \
        +experiment=fhm/lxmert.yaml \
        action=fit \
        datamodule.batch_size=16,32 \
        model.optimizers.0.lr=1e-4,3e-4,2e-5

    python3 main.py --multirun \
        +experiment=$model/t5_classification.yaml \
        action=fit \
        datamodule.batch_size=16 \
        trainer.accumulate_grad_batches=1,2 \
        model.optimizers.0.lr=1e-4,3e-4,2e-5

    python3 main.py --multirun \
        +experiment=$model/t5_classification.yaml \
        action=fit \
        datamodule.batch_size=8 \
        trainer.accumulate_grad_batches=2,4 \
        model.optimizers.0.lr=1e-4,3e-4,2e-5

    python3 main.py --multirun \
        +experiment=$model/bart_classification.yaml \
        action=fit \
        datamodule.batch_size=16 \
        trainer.accumulate_grad_batches=1,2 \
        model.optimizers.0.lr=1e-4,3e-4,2e-5
done