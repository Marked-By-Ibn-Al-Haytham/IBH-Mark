CUDA_VISIBLE_DEVICES=1 python trainer.py \
    --ckpt_path "./exps/nada-swin-encoder-rotateOnly/ckpts" \
    --log_dir "./exps/nada-swin-encoder-rotateOnly/runs" \
    --name "nada-swin-encoder-rotateOnly" \
    --saved_ckpt_path "/home/watermark-m_b/imhotep_checkpoints/imhotep_checkpoints/nada-swin-encoder/model-0162.ckpt" \
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
    --train_path "/home/watermark/data/unsplash_lite_resize256" \
    --eval_path "/home/watermark/data/WatermarkEvaluationDataset-Public/Camera_Capture"\
    --dwt_encoder_arch "swin" \
    --decoder_name "convnext_base" \
    --wandb \
    --wandb_api_key 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 \
    --wandb_project "invismark-watermarking-badran" \
    --wandb_entity="Marked-By-IBH" \

        # --encoder_name "swin" \
    # --decoder_name "convnext_base" \