echo "FLAVA - MAMI - TASK A"

python3 main.py fit \
    --config configs/mami/subtask_a/flava.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2

echo "LXMERT - MAMI - TASK A"

python3 main.py fit \
    --config configs/mami/subtask_a/lxmert_features.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2

echo "VisualBERT - MAMI - TASK A"

python3 main.py fit \
    --config configs/mami/subtask_a/visualbert_features.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2


echo "FLAVA - MAMI - TASK B"

python3 main.py fit \
    --config configs/mami/subtask_b/flava.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2

echo "LXMERT - MAMI - TASK B"

python3 main.py fit \
    --config configs/mami/subtask_b/lxmert_features.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2

echo "VisualBERT - MAMI - TASK B"

python3 main.py fit \
    --config configs/mami/subtask_b/visualbert_features.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2
