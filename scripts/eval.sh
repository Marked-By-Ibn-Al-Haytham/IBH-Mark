#!/bin/bash


python eval.py \
    --ckpt_path "<path/to/ckpt>" \
    --encoder_name "swin" \
    --eval_path "<path/to/eval/data>" \
    --use_dwt \
    --max_batches 100
