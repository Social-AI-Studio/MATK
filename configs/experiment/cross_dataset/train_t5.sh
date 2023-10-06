CUDA_VISIBLE_DEVICES=0 python3 main.py --multirun \
    +experiment=cross_dataset/flant5.yaml \
    action=fit \
    trainer=single_gpu_trainer