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

<img width="1641" height="857" alt="arch_light" src="https://github.com/user-attachments/assets/02249fc9-19d8-457c-a503-c6e8528c1fdb" />



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

- Training script can be found in `scripts/train.sh`, which you can modify with your paths and settings.

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

  
## Trained Model Checkpoint
The trained model checkpoint is available on (https://huggingface.co/MBadran/Marked-By-IBH/tree/main).

## Evaluation

- Evaluation script can be found in `scripts/eval.sh`, which you can modify with your paths and settings.

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


## Watermark Challenge Results

### Embedding Distortion Performance

| Metric | Camera Capture | Synthetic |
|--------|:--------------:|:---------:|
| PSNR | 58.54 | 58.69 |
| wPSNR | 53.69 | 53.83 |
| SSIM | 0.9996 | 0.9995 |
| JND Pass Rate | 1.0000 | 1.0000 |
| FID | 0.0695 | 0.0752 |

---

### Robustness Evaluation (BER)

#### Signal Processing Attacks

| Attack | Camera Capture | Synthetic |
|--------|:--------------:|:---------:|
| Gaussian Noise | 0.0715 | 0.0970 |
| Speckle Noise | 0.0671 | 0.0777 |
| Blurring | 0.0282 | 0.0253 |
| Brightness Adjustment | 0.0490 | 0.0435 |
| Sharpness Adjustment | 0.0294 | 0.0267 |
| Gamma Correction | 0.0417 | 0.0317 |
| Histogram Equalization | 0.0240 | 0.0177 |
| Median Filtering | 0.0327 | 0.0300 |

#### Geometric Attacks

| Attack | Camera Capture | Synthetic |
|--------|:--------------:|:---------:|
| Rotation | 0.0345	| 0.0344  |
| Resizing | 0.0278 | 0.0256 |
| Scaling | 0.0288 | 0.0264 |
| Cropping | 0.1056 | 0.0871 |
| Flipping | 0.0561 | 0.0443 |

#### Compression Attacks

| Attack | Camera Capture | Synthetic |
|--------|:--------------:|:---------:|
| JPEG | 0.1252 | 0.1880 |
| JPEG 2000 | 0.0393 | 0.0428 |
| JPEG XS | 0.3174 | 0.2891 |
| JPEG XL | 0.1626 | 0.2612 |
| JPEG AI | 0.4074 | 0.4353 |

#### Generative AI Manipulation Attacks

| Attack | Camera Capture | Synthetic |
|--------|:--------------:|:---------:|
| AI-Manipulation (Remove) | 0.1146 | 0.1441 |


## Acknowledgements:

Marked by IBH is built upon [InvisMark](https://arxiv.org/abs/2411.07795) as the baseline and adapts the pipeline to support wavelet-domain watermarking. 

<hr>
