CUDA_VISIBLE_DEVICES=1 python3 main.py \
    +experiment=latent_hatred/llava_clm.yaml \
    action=fit \
    model.save_dir=test \
    trainer=debug_trainer 