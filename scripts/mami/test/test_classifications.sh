echo "FLAVA - MAMI"

python3 main.py fit \
    --config configs/mami/flava.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2

echo "LXMERT - MAMI"

python3 main.py fit \
    --config configs/mami/lxmert_features.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2

echo "VisualBERT - MAMI"

python3 main.py fit \
    --config configs/mami/visualbert_features.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2

echo "BART - MAMI"

python3 main.py fit \
    --config configs/mami/bart.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2

echo "T5 - MAMI"

python3 main.py fit \
    --config configs/mami/t5.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2

