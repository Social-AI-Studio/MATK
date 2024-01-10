python3 main.py \
    +experiment=latent_hatred/t5_clm.yaml \
    action=fit \
    trainer=single_gpu_trainer

# python3 main.py \
#     +experiment=latent_hatred/t5_clm.yaml \
#     action=test \
#     +model_checkpoint=experiments/baseline/sbic/t5-clm/best.ckpt \
#     trainer=single_gpu_trainer