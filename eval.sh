#!/bin/bash
#  eval.py eval_aiAttacks.py

CUDA_VISIBLE_DEVICES=1 python eval_aiAttacks.py \
    --ckpt_path "/home/watermark-m_b/imhotep_checkpoints/imhotep_checkpoints/nada-swin-encoder/model-0162.ckpt" \
    --encoder_name "swin" \
    --decoder_name "convnext_base" \
    --eval_path "/home/watermark/data/WatermarkEvaluationDataset-Public/Synthetic" \
    --use_dwt \
    --max_batches 10

        # --eval_path "/home/watermark/data/WatermarkEvaluationDataset-Public/ Camera_Capture " Synthetic  Camera_Capture_resize256 Synthetic_resize256
