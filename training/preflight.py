"""Fail before a costly run if data, GPU, disk, model access or adapter APIs are missing."""
import argparse
import json
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download, model_info, try_to_load_from_cache
from diffusers import Flux3Transformer2DModel, ModularPipeline
from diffusers.loaders import Flux3LoraLoaderMixin

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--include-evaluation", action="store_true")
args = parser.parse_args()
dataset = root / "dataset"
if not dataset.exists():
    dataset = root.parent / "references/dataset"
assert torch.cuda.is_available(), "CUDA is required"
assert torch.cuda.get_device_properties(0).total_memory >= 45 * 1024**3, "Expected a GPU with >=45 GiB VRAM"
data = load_dataset("imagefolder", data_dir=str(dataset))["train"]
assert len(data) == 20
assert all(x.endswith("thlpvk style") for x in data["text"])
assert all(im.width * im.height >= 1_000_000 for im in data["image"])
repositories = [
    ("black-forest-labs/flux-3-image-klein-base", "3f7fa5ee22ca1d5a5cc095ac7cc96e6a173b4c86"),
]
if args.include_evaluation:
    repositories.append(("black-forest-labs/flux-3-image-klein", None))
for repo, revision in repositories:
    hf_hub_download(repo, "text_encoder/config.json", revision=revision)
    info = model_info(repo, revision=revision, files_metadata=True)
    missing_bytes = 0
    for item in info.siblings:
        if item.rfilename.startswith(("text_encoder/", "transformer/", "vae/")) and item.rfilename.endswith(".safetensors"):
            cached = try_to_load_from_cache(repo, item.rfilename, revision=revision or info.sha)
            if not isinstance(cached, str) or not Path(cached).is_file():
                missing_bytes += item.size or 0
    available = shutil.disk_usage(root).free
    assert available >= missing_bytes + 12 * 1024**3, (
        f"Need {(missing_bytes / 1024**3):.1f} GiB for uncached weights plus 12 GiB for checkpoints; "
        f"{available / 1024**3:.1f} GiB available"
    )
assert hasattr(Flux3LoraLoaderMixin, "load_lora_weights")
print(json.dumps({"status": "passed", "images": len(data), "gpu": torch.cuda.get_device_name(0)}, indent=2))
