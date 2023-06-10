echo "BART - FHM"

python3 main.py fit \
    --config configs/fhm/normal/bart.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2

echo "T5 - FHM"

python3 main.py fit \
    --config configs/fhm/normal/t5.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2  \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2

python3 main.py fit \
    --config configs/fhm/normal/roberta_base.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2  \
    --trainer.limit_train_batches 2 \
    --trainer.limit_val_batches 2

python3 main.py fit \
    --config configs/fhm/normal/roberta_prompt.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2  \
    --trainer.limit_train_batches 2 \
    --trainer.limit_val_batches 2