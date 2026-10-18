#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
# Run only on the approved CUDA host. Keep credentials outside this folder.
python3 -c 'import torch; assert torch.cuda.is_available(), "A CUDA GPU is required"; print(torch.cuda.get_device_name(0))'
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'diffusers @ git+https://github.com/huggingface/diffusers.git@b800f828f3951c4d206e154411bcbfa693cd322e' \
  'accelerate>=0.31.0' 'transformers>=4.57.0' 'peft>=0.11.1' 'datasets>=3.1.0' \
  ftfy tensorboard Jinja2 sentencepiece bitsandbytes prodigyopt torchvision
python patch_trainer.py
python -m pip freeze > environment-lock.txt
python preflight.py
