python3 main.py fit \
    --config configs/harmeme/intensity/bart.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2


python3 main.py fit \
    --config configs/harmeme/intensity/t5.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2  \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2

python3 main.py fit \
    --config configs/harmeme/target/bart.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2


python3 main.py fit \
    --config configs/harmeme/target/t5.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2  \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2

python3 main.py fit \
    --config configs/harmeme/intensity/roberta_base.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2  \
    --trainer.limit_train_batches 2 \
    --trainer.limit_val_batches 2

python3 main.py fit \
    --config configs/harmeme/intensity/roberta_prompt.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2  \
    --trainer.limit_train_batches 2 \
    --trainer.limit_val_batches 2