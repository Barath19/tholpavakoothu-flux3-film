"""Launch a bounded training process after cloud/access/eligibility approvals."""
import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--max-minutes", type=int, default=150)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
assert 1 <= args.steps <= 2000 and 1 <= args.max_minutes <= 180
dataset = root / "dataset"
if not dataset.exists():
    dataset = root.parent / "references/dataset"
prompts = json.loads((root / "evaluation-prompts.json").read_text())
cmd = [sys.executable, "-m", "accelerate.commands.launch", "--num_processes=1", "--num_machines=1",
       "--mixed_precision=bf16", "--dynamo_backend=no", str(root / "train_flux3.py"),
       "--pretrained_model_name_or_path=black-forest-labs/flux-3-image-klein-base",
       "--revision=3f7fa5ee22ca1d5a5cc095ac7cc96e6a173b4c86",
       f"--dataset_name={dataset}", "--caption_column=text", "--image_column=image",
       "--instance_prompt=thlpvk style", f"--output_dir={root / 'output'}",
       "--mixed_precision=bf16", "--resolution=1024", "--use_aspect_ratio_buckets", "--center_crop",
       "--train_batch_size=1", "--gradient_accumulation_steps=1", "--gradient_checkpointing",
       "--cache_latents", "--offload", "--rank=32", "--lora_alpha=32", "--optimizer=adamW",
       "--use_8bit_adam", "--learning_rate=1e-4", "--adam_beta1=0.9", "--adam_beta2=0.95",
       "--lr_scheduler=constant", "--lr_warmup_steps=0", f"--max_train_steps={args.steps}",
       "--checkpointing_steps=250", "--num_validation_images=1", "--report_to=tensorboard",
       "--seed=1701", "--skip_final_inference", f"--validation_prompt={prompts['cases'][0]['expanded']}"]
if args.resume:
    cmd += ["--resume_from_checkpoint=latest"]
if args.dry_run:
    print(json.dumps(cmd, indent=2))
    raise SystemExit(0)
subprocess.run([sys.executable, str(root / "preflight.py")], check=True)
(root / "output").mkdir(exist_ok=True)
(root / "output/run-config.json").write_text(json.dumps({"command": cmd, "max_minutes": args.max_minutes}, indent=2))
with (root / "output/train.log").open("a") as log:
    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=root, start_new_session=True)
    try:
        result = process.wait(timeout=args.max_minutes * 60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        print("Training time limit reached. Saved checkpoints remain available. Stop the cloud Devlab to end billing.")
        raise SystemExit(124)
    if result:
        raise SystemExit(result)
print("Training finished. Review checkpoints, run evaluate.py, retrieve outputs, and stop the cloud Devlab.")
