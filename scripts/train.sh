#!/bin/bash

python trainer.py \
    --ckpt_path "./exps/dwt/ckpts" \
    --log_dir "./exps/dwt/runs" \
    --name "dwt" \
    --batch_size 32 \
    --lr 0.0002 \
    --num_epochs 500 \
    --num_bits 100 \
    --image_size 256 \
    --beta_max 40.0 \
    --beta_epochs 20 \
    --num_noises 2 \
    --noise_start_epoch 20 \
    --use_dwt \
    --train_path "<path/to/train/data>" \
    --eval_path "<path/to/eval/data>" \
    --wandb \
    --wandb_api_key <YOUR_WANDB_API_KEY> \
    --wandb_project "marked-by-ibh" \
    --dwt_encoder_arch "swin"