#!/usr/bin/env python3
"""Crop generated images per-LoRA by combining background and foreground flows.

The script expects filenames that follow either of the conventions
``character_[1-3]_LoRA1_{1,2}_LoRA2_{1,2}_..._X_seedY.ext`` or
``LoRA1_{1,2}_LoRA2_{1,2}_..._X_seedY.ext`` (no character prefix). Each LoRA
token is written as ``{type}_{index}``.  Images are processed in alphabetical
order and the outputs are written as ``Z_X.ext`` where ``Z`` is the (1-based)
order of the input image in that sorted list and ``X`` is copied from the source
filename.

For every image we:

* Parse all LoRA names that appear between the ``character_*`` prefix and the
  ``_X_seed`` suffix.
* Route background LoRAs (`background_{1,2}`) through the background-blur
  pipeline, foreground LoRAs (`clothing_{1,2}`, `object_{1,2}`) through the crop
  pipeline, and style LoRAs (`style_{1,2}`) simply copy the untouched image to
  their respective folders.
* Save results inside ``<output_root>/<lora_name>/Z_X.ext``.  Sub-folders are
  created on-demand.

By default background LoRAs run with the negative prompts listed in
``negative_foreground_prompts.txt``. Foreground prompts use the fixed set
`clothing_{1,2}` → “school uniform”, `object_1` → “umbrella”, `object_2` →
“bubble”, and otherwise fall back to the LoRA name with underscores replaced by
spaces.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FILENAME_RE = re.compile(
    r"^(?:character_(?P<char_id>[1-3])_)?(?P<lora_block>.+)_(?P<x>\d+)_seed(?P<seed>\d+)$",
    re.IGNORECASE,
)
BACKGROUND_LORAS = {"background_1", "background_2"}
FOREGROUND_LORAS = {"clothing_1", "clothing_2", "object_1", "object_2"}
STYLE_LORAS = {"style_1", "style_2"}
DEFAULT_FOREGROUND_PROMPT_MAP = {
    "clothing_1": "school uniform",
    "clothing_2": "school uniform",
    "object_1": "umbrella",
    "object_2": "bubble",
}


def blur_foreground_with_gaussian(
    img: np.ndarray,
    fg_mask: np.ndarray,
    ksize: int,
) -> np.ndarray:
    """Keep background untouched while blurring the segmented foreground."""
    if ksize % 2 == 0:
        raise ValueError("ksize must be odd")

    fg_mask = np.clip(fg_mask.astype(np.float32), 0.0, 1.0)
    img_f = img.astype(np.float32)

    blurred = cv2.GaussianBlur(img_f, (ksize, ksize), 0)
    fg_mask_3d = fg_mask[..., None]

    out = img_f * (1.0 - fg_mask_3d) + blurred * fg_mask_3d
    return out.clip(0, 255).astype(np.uint8)


def fill_background_with_local_background_mean(
    img: np.ndarray,
    fg_mask: np.ndarray,
    ksize: int,
) -> np.ndarray:
    """Keep foreground untouched and smooth the background with its local mean."""
    if ksize % 2 == 0:
        raise ValueError("ksize must be odd")

    fg_mask = fg_mask.astype(np.float32)
    bg_mask = 1.0 - fg_mask

    img_f = img.astype(np.float32)
    bg_img = img_f * bg_mask[..., None]

    sum_bg = cv2.blur(bg_img, (ksize, ksize))
    count_bg = cv2.blur(bg_mask, (ksize, ksize))
    count_bg = np.clip(count_bg, 1e-6, None)[..., None]

    local_bg_mean = sum_bg / count_bg
    out = img_f.copy()

    bg_mask_bool = bg_mask.astype(bool)
    out[bg_mask_bool] = local_bg_mean[bg_mask_bool]

    return out.clip(0, 255).astype(np.uint8)


def to_numpy(array) -> np.ndarray:
    """Convert torch.Tensor, PIL Image, or numpy input to numpy array."""
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    elif isinstance(array, Image.Image):
        array = np.asarray(array)
    return np.asarray(array)


def iter_image_paths(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_prompts(prompt_file: Path) -> list[str]:
    if prompt_file is None:
        return []

    if not prompt_file.exists() or not prompt_file.is_file():
        raise FileNotFoundError(f"Prompt file does not exist: {prompt_file}")

    prompts = [line.strip() for line in prompt_file.read_text().splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_file}")
    return prompts


def parse_lora_names(filename_stem: str) -> tuple[list[str], str]:
    match = FILENAME_RE.match(filename_stem)
    if not match:
        raise ValueError(
            "Filename must match [character_[1-3]_]<lora>..._X_seedY.*; "
            f"got '{filename_stem}'.",
        )

    lora_block = match.group("lora_block")
    tokens = lora_block.split("_")
    if len(tokens) < 2 or len(tokens) % 2 != 0:
        raise ValueError(
            f"LoRA portion '{lora_block}' must contain pairs like name_index.",
        )

    lora_names = ["_".join(tokens[i : i + 2]) for i in range(0, len(tokens), 2)]
    return lora_names, match.group("x")


def combine_masks_from_prompts(
    processor: Sam3Processor,
    state,
    prompts: Sequence[str],
    mask_threshold: float,
) -> np.ndarray | None:
    combined_mask: np.ndarray | None = None

    for prompt in prompts:
        prompt = prompt.strip()
        if not prompt:
            continue

        output = processor.set_text_prompt(state=state, prompt=prompt)
        masks = output.get("masks")
        if masks is None or len(masks) == 0:
            continue

        for mask in masks:
            mask_np = to_numpy(mask)
            if mask_np.ndim == 3:
                mask_np = np.squeeze(mask_np, axis=0)

            mask_bool = mask_np > mask_threshold
            combined_mask = mask_bool if combined_mask is None else np.logical_or(combined_mask, mask_bool)

    return combined_mask


def ensure_subfolder(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_foreground_prompt(lora_name: str) -> str:
    return DEFAULT_FOREGROUND_PROMPT_MAP.get(lora_name, lora_name.replace("_", " "))


def crop_foreground(
    image_np: np.ndarray,
    fg_mask: np.ndarray,
    ksize: int,
) -> np.ndarray | None:
    if not fg_mask.any():
        return None

    bg_filled = fill_background_with_local_background_mean(
        image_np,
        fg_mask.astype(np.float32),
        ksize,
    )

    ys, xs = np.where(fg_mask)
    y0_i = int(np.min(ys))
    y1_i = int(np.max(ys) + 1)
    x0_i = int(np.min(xs))
    x1_i = int(np.max(xs) + 1)

    img_h, img_w = fg_mask.shape
    x0_i = max(0, min(img_w, x0_i))
    x1_i = max(0, min(img_w, x1_i))
    y0_i = max(0, min(img_h, y0_i))
    y1_i = max(0, min(img_h, y1_i))

    if x1_i <= x0_i or y1_i <= y0_i:
        return None

    return bg_filled[y0_i:y1_i, x0_i:x1_i]


def save_image(array: np.ndarray, dest: Path) -> None:
    Image.fromarray(array).save(dest)
    print(f"[INFO] Saved {dest}")


def build_processor() -> Sam3Processor:
    model = build_sam3_image_model()
    model.eval()
    return Sam3Processor(model)


def parse_args() -> argparse.Namespace:
    default_prompt_file = Path(__file__).resolve().with_name("negative_foreground_prompts.txt")

    parser = argparse.ArgumentParser(
        description="Route images through background/foreground cropping per LoRA.",
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="Folder with the generated images.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Output root folder. Defaults to <input_dir>_cropped_by_lora.",
    )
    parser.add_argument(
        "--background_prompt_file",
        type=Path,
        default=default_prompt_file,
        help=f"Text file with one prompt per line (default: {default_prompt_file.name}).",
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.5,
        help="Probability threshold to binarize the segmentation mask (default: 0.5).",
    )
    parser.add_argument(
        "--ksize",
        type=int,
        default=71,
        help="Odd kernel size used in both background blur and background filling (default: 71).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir = args.output_dir or input_dir.with_name(f"{input_dir.name}_cropped_by_lora")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        background_prompts = load_prompts(args.background_prompt_file) if args.background_prompt_file else []
    except (FileNotFoundError, ValueError) as exc:
        print(f"[WARN] {exc}; background LoRAs will save original images.")
        background_prompts = []

    processor = build_processor()
    images = list(iter_image_paths(input_dir))
    if not images:
        print(f"[WARN] No image files found in {input_dir}")
        return

    for order_idx, image_path in enumerate(images, start=1):
        lora_names: list[str]
        sample_idx: str
        try:
            lora_names, sample_idx = parse_lora_names(image_path.stem)
        except ValueError as exc:
            print(f"[WARN] {exc} Skipping {image_path.name}.")
            continue

        background_targets: list[str] = []
        foreground_targets: list[str] = []
        style_targets: list[str] = []
        skipped_targets: list[str] = []

        for lora_name in lora_names:
            if lora_name in BACKGROUND_LORAS:
                background_targets.append(lora_name)
            elif lora_name in STYLE_LORAS:
                style_targets.append(lora_name)
            elif lora_name in FOREGROUND_LORAS:
                foreground_targets.append(lora_name)
            else:
                skipped_targets.append(lora_name)
        if skipped_targets:
            print(
                f"[WARN] Unhandled LoRA types for {image_path.name}: "
                f"{', '.join(skipped_targets)}.",
            )

        if not background_targets and not foreground_targets and not style_targets:
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:  # pragma: no cover
            print(f"[ERROR] Failed to open {image_path.name}: {exc}")
            continue

        image_np = np.asarray(image)

        with torch.no_grad():
            state = processor.set_image(image)

        suffix = image_path.suffix.lower()
        filename = f"{order_idx}_{sample_idx}{suffix}"

        if background_targets:
            if background_prompts:
                with torch.no_grad():
                    combined_mask = combine_masks_from_prompts(
                        processor=processor,
                        state=state,
                        prompts=background_prompts,
                        mask_threshold=args.mask_threshold,
                    )
                if combined_mask is None or not combined_mask.any():
                    print(
                        f"[WARN] Background prompts produced no mask for {image_path.name}; "
                        "saving original image.",
                    )
                    background_result = image_np
                else:
                    background_result = blur_foreground_with_gaussian(
                        image_np,
                        combined_mask.astype(np.float32),
                        args.ksize,
                    )
            else:
                background_result = image_np

            for lora_name in background_targets:
                dest = ensure_subfolder(output_dir, lora_name) / filename
                save_image(background_result, dest)

        for lora_name in style_targets:
            dest = ensure_subfolder(output_dir, lora_name) / filename
            save_image(image_np, dest)

        for lora_name in foreground_targets:
            prompt = resolve_foreground_prompt(lora_name=lora_name)

            with torch.no_grad():
                combined_mask = combine_masks_from_prompts(
                    processor=processor,
                    state=state,
                    prompts=[prompt],
                    mask_threshold=args.mask_threshold,
                )

            if combined_mask is None or not combined_mask.any():
                print(
                    f"[WARN] Foreground prompt '{prompt}' produced no mask for "
                    f"{image_path.name}; saving original image.",
                )
                foreground_result = image_np
            else:
                crop = crop_foreground(
                    image_np=image_np,
                    fg_mask=combined_mask,
                    ksize=args.ksize,
                )
                if crop is None:
                    print(
                        f"[WARN] Invalid crop bounds for {image_path.name}; saving original image.",
                    )
                    foreground_result = image_np
                else:
                    foreground_result = crop

            dest = ensure_subfolder(output_dir, lora_name) / filename
            save_image(foreground_result, dest)


if __name__ == "__main__":
    main()

