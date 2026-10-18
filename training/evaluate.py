"""Paired evaluation: exact same expanded prompt, seed and distilled checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from diffusers import ModularPipeline
from huggingface_hub import model_info

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--scale", type=float, default=0.75)
parser.add_argument("--model", choices=["distilled", "base"], default="distilled")
parser.add_argument("--case", action="append", help="Evaluate only the named frozen prompt case")
parser.add_argument("--revision", help="Pin a verified distilled model revision")
args = parser.parse_args()
assert args.checkpoint.exists()
cases = json.loads((root / "evaluation-prompts.json").read_text())["cases"]
if args.case:
    cases = [case for case in cases if case["id"] in args.case]
    assert len(cases) == len(set(args.case)), "Unknown evaluation case"
repo = "black-forest-labs/flux-3-image-klein" + ("-base" if args.model == "base" else "")
revision = "3f7fa5ee22ca1d5a5cc095ac7cc96e6a173b4c86" if args.model == "base" else (args.revision or model_info(repo).sha)
steps, guidance = (50, 3.5) if args.model == "base" else (16, 1.0)
loaded = ModularPipeline.from_pretrained(repo, revision=revision)
component_names = ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
if args.model == "distilled" and revision == "337337621b8526e120f2275d250242f4a6bedd75":
    shared = ["vae", "text_encoder", "tokenizer"]
    # The encoder/VAE weight hashes were verified identical across both repos.
    cached = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein-base",
        revision="3f7fa5ee22ca1d5a5cc095ac7cc96e6a173b4c86")
    cached.load_components(names=shared,
        torch_dtype={"default": torch.bfloat16, "vae": torch.float32})
    loaded.load_components(names=["transformer", "scheduler"], torch_dtype=torch.bfloat16)
    loaded.update_components(**{name: getattr(cached, name) for name in shared})
    del cached
else:
    loaded.load_components(names=component_names,
        torch_dtype={"default": torch.bfloat16, "vae": torch.float32})
# The public blocks property is a copy; alter it before initializing a new pipeline.
blocks = loaded.blocks
blocks.sub_blocks.pop("prompt_upsampler")
pipe = blocks.init_pipeline()
pipe.update_components(**{name: getattr(loaded, name) for name in component_names})
del loaded
pipe.to("cuda")
out = root / "output/evaluation" / (args.checkpoint.name + "-" + args.model)
out.mkdir(parents=True, exist_ok=True)
results = []
for adapted in [False, True]:
    if adapted:
        pipe.load_lora_weights(str(args.checkpoint), adapter_name="thlpvk")
        pipe.set_adapters("thlpvk", adapter_weights=args.scale)
    for case in cases:
        image = pipe(prompt=case["expanded"], width=1360, height=768, num_inference_steps=steps,
                     guidance_scale=guidance, generator=torch.Generator(device="cuda").manual_seed(case["seed"]),
                     output="images")[0]
        name = f"{case['id']}-{'lora' if adapted else 'baseline'}.png"
        image.save(out / name)
        results.append({**case, "file": name, "adapter": adapted, "scale": args.scale if adapted else 0})
        (out / "manifest.json").write_text(json.dumps({"model": repo, "revision": revision,
            "checkpoint": str(args.checkpoint.resolve()), "steps": steps, "guidance": guidance,
            "purpose": "preliminary base-model preview" if args.model == "base" else "submission comparison",
            "results": results}, indent=2))
print(out)
