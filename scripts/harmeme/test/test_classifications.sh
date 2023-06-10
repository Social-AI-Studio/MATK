### HarMemes Intensity Classification ###
echo "FLAVA - HarMemes Intensity"

python3 main.py fit \
    --config configs/harmeme/intensity/flava.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2

echo "LXMERT - HarMemes Intensity"

python3 main.py fit \
    --config configs/harmeme/intensity/lxmert_features.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2

echo "VisualBERT - HarMemes Intensity"

python3 main.py fit \
    --config configs/harmeme/intensity/visualbert_features.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2

echo "FLAVA - HarMemes Target"

## HarMemes Target Classification ###
python3 main.py fit \
    --config configs/harmeme/target/flava.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2

echo "LXMERT - HarMemes Target"

python3 main.py fit \
    --config configs/harmeme/target/lxmert_features.yaml \
    --seed_everything 1111 \
    --data.batch_size 2 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2

echo "VisualBERT - HarMemes Target"

python3 main.py fit \
    --config configs/harmeme/target/visualbert_features.yaml \
    --seed_everything 1111 \
    --data.batch_size 2 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2