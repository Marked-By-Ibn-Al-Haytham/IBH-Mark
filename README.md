# Marked by IBH

<img width="1672" height="941" alt="ChatGPT Image Apr 22, 2026, 05_48_20 PM(1)" src="https://github.com/user-attachments/assets/66346875-012e-463b-994d-d9b4e9d82e7f" />


## Overview

Marked by IBH is a robust DWT-based watermarking system against attacks and manipulations extending InvisMark.

The method:

- embeds a binary watermark into an image
- keeps the visual distortion small
- recovers the watermark after image attacks

The repo supports 3 modes:

- `DCT`
- `DWT`
- `DCT + DWT`

Main entrypoints:

- `trainer.py`: train
- `eval.py`: evaluate
- `eval_aiAttacks.py`: evaluate with AI-style attacks

## Architecture



## Installation

```bash
python create -n marked-by-ibh python=3.10
conda activate marked-by-ibh
pip install -r requirements.txt
```

```bash
pip install "watermarkbench[all] @ git+https://github.com/JPEG-Trust-Community/watermarking.git#subdirectory=evaluation_metric/package"
```

```bash
cd Swin_Unet
python download_swin_ckpt.py
```


## Dataset Format

The code uses `torchvision.datasets.ImageFolder`, so your data must look like this:

```text
data/
  train/
    images/
      img1.png
      img2.png
  eval/
    images/
      img1.png
      img2.png
```

Use:

- `--train_path /path/to/data/train`
- `--eval_path /path/to/data/eval`

## Configuration

Most runs only need these arguments:

- `--train_path`
- `--eval_path`
- `--ckpt_path`
- `--log_dir`
- `--num_epochs`
- `--batch_size`
- `--lr`
- `--num_bits`
- `--image_size`

Mode flags:

- `--use_dct`
- `--use_dwt`
- `--use_dct_dwt`

DWT encoder choices:

- `unet`
- `swin`
- `convnext`
- `resnet50`

## Training

Example:

```bash
python trainer.py \
  --train_path /path/to/data/train \
  --eval_path /path/to/data/eval \
  --ckpt_path ./exps/run1/ckpts \
  --log_dir ./exps/run1/runs \
  --name run1 \
  --batch_size 32 \
  --lr 2e-4 \
  --num_epochs 500 \
  --num_bits 100 \
  --image_size 256 \
  --beta_max 40.0 \
  --beta_epochs 20 \
  --num_noises 2 \
  --noise_start_epoch 20 \
  --use_dwt \
  --dwt_encoder_arch swin \
```

Switch the mode with one of:

- `--use_dct`
- `--use_dwt`
- `--use_dct_dwt`

Resume training:

```bash
python trainer.py \
  --train_path /path/to/data/train \
  --eval_path /path/to/data/eval \
  --ckpt_path ./exps/run1/ckpts \
  --log_dir ./exps/run1/runs \
  --saved_ckpt_path ./exps/run1/ckpts/model-0123.ckpt \
  --use_dwt
```

## Evaluation

Standard evaluation:

```bash
python eval.py \
  --ckpt_path ./exps/run1/ckpts/model-0499.ckpt \
  --encoder_name swin \
  --eval_path /path/to/data/eval \
  --image_size 256 \
  --num_bits 100 \
  --max_batches 100 \
  --use_dwt
```

AI-attack evaluation:

```bash
python eval_aiAttacks.py \
  --ckpt_path ./exps/run1/ckpts/model-0499.ckpt \
  --encoder_name swin \
  --decoder_name convnext_base \
  --eval_path /path/to/data/eval \
  --image_size 256 \
  --num_bits 100 \
  --max_batches 100 \
  --use_dwt
```

## Outputs

Training writes:

- checkpoints to `--ckpt_path`
- TensorBoard logs to `--log_dir`
- optional WandB logs if enabled

View logs:

You can view logs from WandB or TensorBoard. 

For TensorBoard:

```bash
tensorboard --logdir ./exps
```

For WandB, use `--wandb` flag in the train.sh, add the `--wandb_api_key` and `--wandb_project` for the project name. Logs will be uploaded to WandB.

## Acknowledgements:

Marked by IBH is built upon [InvisMark](https://arxiv.org/abs/2411.07795) as the baseline and adapts the pipeline to support wavelet-domain watermarking. 

<hr>
