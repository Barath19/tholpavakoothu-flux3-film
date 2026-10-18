# Jatayu: The Last Flight

A 90-second Ramayana story in Kerala’s Tholpavakoothu shadow-puppet style, with original music, sound effects and English subtitles.

**[Watch the film](https://huggingface.co/Barath/tholpavakoothu-flux3-lora/resolve/main/films/jatayu/jatayu-the-last-flight.mp4)** · **[LoRA on Hugging Face](https://huggingface.co/Barath/tholpavakoothu-flux3-lora)**

[Clean film](https://huggingface.co/Barath/tholpavakoothu-flux3-lora/resolve/main/films/jatayu/jatayu-the-last-flight-clean.mp4) · [Subtitles](films/jatayu/narration.srt) · [Storyboard](films/jatayu/storyboard.jpg) · [Film credits](films/jatayu/FILM-CREDITS.txt)

**Style:** Kerala leather shadow-puppet performance: dark cutouts, perforations and warm cloth backlighting.<br>
**Data:** 20 credited Wikimedia Commons photographs: 17 CC BY-SA 4.0 and 3 CC BY 4.0. [Sources and credits](CREDITS.txt).<br>
**Checkpoint:** Step 2000, rank-32 LoRA trained on FLUX 3 Image Klein Base; images generated with distilled Klein.

## Six training images

[Full dataset](https://huggingface.co/Barath/tholpavakoothu-flux3-lora/resolve/main/tholpavakoothu-dataset.zip) · [Image sources and preparation](training-examples/sources.json)

| 01 | 06 | 11 |
|---|---|---|
| ![Training image 01](training-examples/01.jpg) | ![Training image 06](training-examples/06.jpg) | ![Training image 11](training-examples/11.jpg) |
| **12** | **16** | **20** |
| ![Training image 12](training-examples/12.jpg) | ![Training image 16](training-examples/16.jpg) | ![Training image 20](training-examples/20.jpg) |

## Base vs. LoRA

Same prompt and seed **1711**, distilled Klein, 16 steps. Adapter strength: **0 → 0.75**. [Prompt and settings](comparison/manifest.json).

| Base | LoRA |
|---|---|
| ![Base output](comparison/fisherman-and-boat-baseline.png) | ![LoRA output](comparison/fisherman-and-boat-lora.png) |
