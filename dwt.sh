#!/bin/bash

python trainer.py \
    --ckpt_path "./exps/dwt-E2E-swin-vitl/ckpts" \
    --log_dir "./exps/dwt-E2E-swin-vitl/runs" \
    --name "dwt-E2E-swin-vitl" \
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
    --train_path "/scratch/dr/m.badran/watermark/dataset/unsplash_lite_resize256" \
    --eval_path "/scratch/dr/m.badran/watermark/dataset/WatermarkEvaluationDataset-Public/Camera_Capture" \
    --wandb \
    --wandb_api_key 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 \
    --wandb_project "invismark" \
    --wandb_entity "Marked-By-IBH" \
    --dwt_encoder_arch "swin" \
    --decoder_name "vit_l_16"