# SAM 3 concept cropping for multi-LoRA composition

This repository crops generated images for [Training-Free Multi-Concept LoRA Composition with Prompt-Aware Weighting](https://arxiv.org/abs/2606.03792). Faces are cropped in [Prompt-Aware-Multi-LoRA-Composition](https://github.com/GeorgeTsoumplekas/PromptAware-Multi-LoRA-Composition) with a face detector. Every other concept, such as clothing, objects, backgrounds, and styles, is cropped here with [SAM 3](https://github.com/facebookresearch/sam3).

SAM 3 segments a text prompt, then this repo turns that mask into a crop:

| LoRA ID | What is saved |
| --- | --- |
| `clothing_1`, `clothing_2` | Tight crop of the segmented school uniform. |
| `object_1` | Tight crop of the umbrella. |
| `object_2` | Tight crop of the bubble. |
| `background_1`, `background_2` | Full image with the foreground blurred. |
| `style_1`, `style_2` | The original image, copied unchanged. Style has no region to segment. |
| `character_1`, `character_2`, `character_3` | Skipped. These are cropped in the original paper's repository. |

The `sam3/` package is the SAM 3 image model used for these crops. It is covered by the [SAM License](LICENSE).

## Installation

You need a CUDA GPU and CUDA 12.6 or newer. From this repository:

```bash
conda create -n sam3 python=3.12 -y
conda activate sam3
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

## Download the SAM 3 checkpoint

SAM 3 weights are gated. Request access on the [facebook/sam3](https://huggingface.co/facebook/sam3) Hugging Face repository. After it is granted, create a token and log in:

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
```

## Crop images

Image names follow the multi-lora composition repository:

```text
<lora_id>_<lora_id>_..._<index>_seed<seed>.png
```

For example, `character_2_clothing_2_object_1_1_seed42.png`. Character ids are not cropped here.

Activate the environment, then crop one folder:

```bash
conda activate sam3
bash crop_images.sh /path/to/images /path/to/images_cropped
```

A parent folder of method directories is handled the same way. This matches `outputs/generated/` from `scripts/generate_benchmark.sh` in the composition repository:

```bash
bash crop_images.sh \
  /path/to/Prompt-Aware-Multi-LoRA-Composition/outputs/generated \
  /path/to/Prompt-Aware-Multi-LoRA-Composition/outputs/cropped
```

Background prompts used for the blur step are listed in `negative_foreground_prompts.txt`, one prompt per line.

To call the Python script on a single folder directly:

```bash
python crop_by_lora_type.py \
  --input_dir /path/to/images \
  --output_dir /path/to/images_cropped \
  --background_prompt_file negative_foreground_prompts.txt
```

## Citation

If you use the cropping pipeline, please cite the composition paper and SAM 3.

```bibtex
@inproceedings{tsoumplekas2026promptaware,
  title     = {Training-Free Multi-Concept LoRA Composition with Prompt-Aware Weighting},
  author    = {Tsoumplekas, Georgios and Bounareli, Stella and Argyriou, Vasileios},
  booktitle = {IEEE International Conference on Automatic Face and Gesture Recognition (FG)},
  year      = {2026},
  url       = {https://ieeexplore.ieee.org/document/11557018}
}
```

```bibtex
@misc{carion2025sam3segmentconcepts,
  title         = {SAM 3: Segment Anything with Concepts},
  author        = {Carion, Nicolas and Gustafson, Laura and Hu, Yuan-Ting and Debnath, Shoubhik and Hu, Ronghang and Suris, Didac and Ryali, Chaitanya and Alwala, Kalyan Vasudev and Khedr, Haitham and Huang, Andrew and Lei, Jie and Ma, Tengyu and Guo, Baishan and Kalla, Arpit and Marks, Markus and Greer, Joseph and Wang, Meng and Sun, Peize and R{\"a}dle, Roman and Afouras, Triantafyllos and Mavroudi, Effrosyni and Xu, Katherine and Wu, Tsung-Han and Zhou, Yu and Momeni, Liliane and Hazra, Rishi and Ding, Shuangrui and Vaze, Sagar and Porcher, Francois and Li, Feng and Li, Siyuan and Kamath, Aishwarya and Cheng, Ho Kei and Doll{\'a}r, Piotr and Ravi, Nikhila and Saenko, Kate and Zhang, Pengchuan and Feichtenhofer, Christoph},
  year          = {2025},
  eprint        = {2511.16719},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2511.16719}
}
```
