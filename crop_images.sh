#!/usr/bin/env bash
# Crop a folder of composed images into one directory per LoRA.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CROP_SCRIPT="${SCRIPT_DIR}/crop_by_lora_type.py"
PROMPTS_FILE="${SCRIPT_DIR}/negative_foreground_prompts.txt"

usage() {
  cat <<'EOF'
Usage: crop_images.sh INPUT_DIR [OUTPUT_DIR]

Crop generated images into one folder per LoRA id.

INPUT_DIR may be either:
  - a folder of images, or
  - a folder whose immediate subfolders each contain images
    (the layout written by Prompt-Aware-Multi-LoRA-Composition).

OUTPUT_DIR defaults to <INPUT_DIR>_cropped.
When INPUT_DIR contains subfolders, each one is written to
<OUTPUT_DIR>/<subfolder>_cropped.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 1
fi

INPUT_DIR="$(realpath "$1")"
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "[ERROR] Input directory not found: $INPUT_DIR" >&2
  exit 1
fi

if [[ ! -f "$CROP_SCRIPT" ]]; then
  echo "[ERROR] Unable to locate crop_by_lora_type.py in $SCRIPT_DIR" >&2
  exit 1
fi

if [[ ! -f "$PROMPTS_FILE" ]]; then
  echo "[ERROR] negative_foreground_prompts.txt not found in $SCRIPT_DIR" >&2
  exit 1
fi

if [[ $# -eq 2 ]]; then
  OUTPUT_DIR="$(realpath -m "$2")"
else
  OUTPUT_DIR="$(realpath -m "${INPUT_DIR}_cropped")"
fi

has_images() {
  local dir="$1"
  find "$dir" -mindepth 1 -maxdepth 1 -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \) \
    -print -quit | grep -q .
}

crop_one() {
  local src="$1"
  local dest="$2"
  echo "[INFO] Cropping $src -> $dest"
  mkdir -p "$dest"
  "$PYTHON_BIN" "$CROP_SCRIPT" \
    --input_dir "$src" \
    --output_dir "$dest" \
    --background_prompt_file "$PROMPTS_FILE"
}

if has_images "$INPUT_DIR"; then
  crop_one "$INPUT_DIR" "$OUTPUT_DIR"
  exit 0
fi

mapfile -d '' subdirs < <(find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z) || true
if [[ ${#subdirs[@]} -eq 0 ]]; then
  echo "[WARN] No images or subfolders found in $INPUT_DIR."
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
overall_status=0
for subdir in "${subdirs[@]}"; do
  name="$(basename "$subdir")"
  if ! crop_one "$subdir" "$OUTPUT_DIR/${name}_cropped"; then
    echo "[ERROR] Cropping failed for $subdir" >&2
    overall_status=1
  fi
done

exit "$overall_status"
