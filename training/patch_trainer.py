"""Apply narrow, asserted adaptations to the pinned upstream trainer."""
from pathlib import Path
import difflib

root = Path(__file__).resolve().parent
original = (root / "vendor/train_flux3_original.py").read_text()
patched = original
assert patched.count('subfolder="tokenizer"') == 2
patched = patched.replace('subfolder="tokenizer"', 'subfolder="text_encoder"')
needle = '    pipeline = Flux3AutoBlocks().init_pipeline()'
assert patched.count(needle) == 1
patched = patched.replace(needle, '''    blocks = Flux3AutoBlocks()
    # Prompts are already expanded and frozen for reproducible comparisons.
    blocks.sub_blocks.pop("prompt_upsampler")
    pipeline = blocks.init_pipeline()''')
needle = '''            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )'''
assert patched.count(needle) == 1
patched = patched.replace(needle, '''            if os.path.isdir(args.dataset_name):
                dataset = load_dataset("imagefolder", data_dir=args.dataset_name, cache_dir=args.cache_dir)
            else:
                dataset = load_dataset(
                    args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir,
                )''')
patched = patched.replace('"num_inference_steps": 28,', '"num_inference_steps": 50,\n            "guidance_scale": 3.5,')
needle = '                        logger.info(f"Saved state to {save_path}")'
assert patched.count(needle) == 1
patched = patched.replace(needle, needle + '''
                        if args.validation_prompt is not None:
                            transformer_ = unwrap_model(transformer)
                            transformer_.eval()
                            pipeline = make_validation_pipeline(args, transformer_, weight_dtype, accelerator.device)
                            samples = log_validation(
                                pipeline, args, accelerator, validation_pipeline_args,
                                global_step, weight_dtype,
                            )
                            sample_dir = os.path.join(args.output_dir, "samples", f"step-{global_step:04d}")
                            os.makedirs(sample_dir, exist_ok=True)
                            for sample_number, sample in enumerate(samples):
                                sample.save(os.path.join(sample_dir, f"sample-{sample_number:02d}.png"))
                            del pipeline
                            transformer_.train()
                            free_memory()
''')
needle = 'if args.validation_prompt is not None and epoch % args.validation_epochs == 0:'
assert patched.count(needle) == 1
patched = patched.replace(needle, 'if False:  # Validation runs at saved optimizer steps above, not epoch boundaries.')
(root / "train_flux3.py").write_text(patched)
(root / "vendor/trainer.patch").write_text(''.join(difflib.unified_diff(
    original.splitlines(True), patched.splitlines(True),
    fromfile="upstream/train_flux3.py", tofile="training/train_flux3.py")))
print("Patched processor path, dataset loading, pre-expanded prompts, sampler and checkpoint previews.")
