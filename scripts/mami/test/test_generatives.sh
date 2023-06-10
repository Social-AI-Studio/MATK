CUDA_VISIBLE_DEVICES=1 python3 main.py fit \
    --config configs/mami/bart.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2


# python3 main.py fit \
#     --config configs/mami/t5.yaml \
#     --seed_everything 1111 \
#     --trainer.devices 1 \
#     --trainer.max_epochs 2  \
#     --trainer.limit_train_batches 5 \
#     --trainer.limit_val_batches 2