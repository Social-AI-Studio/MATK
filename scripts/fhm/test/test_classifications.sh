echo "### FLAVA - FHM ###"

python3 main.py fit \
    --config configs/fhm/normal/flava.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2

echo "### LXMERT - FHM ###"

python3 main.py fit \
    --config configs/fhm/normal/lxmert_features.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2

echo "### VISUALBERT - FHM ###"

python3 main.py fit \
    --config configs/fhm/normal/visualbert_features.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2