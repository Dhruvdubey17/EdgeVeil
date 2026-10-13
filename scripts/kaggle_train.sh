set -euo pipefail

SOURCE_DIR="${1:?usage: bash kaggle_train.sh <path to the extracted edgeveil source>}"
WORK_DIR=/tmp/edgeveil
OUTPUT_DIR=/kaggle/working
TORCH_BACKEND=cu126
DEVICE_BATCH_SIZE="${EDGEVEIL_DEVICE_BATCH_SIZE:-16}"

export CUDA_VISIBLE_DEVICES=0
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1

rm -rf "$WORK_DIR"
cp -r "$SOURCE_DIR" "$WORK_DIR"
cd "$WORK_DIR"

pip install --quiet uv
uv sync --locked --extra train --no-dev
TORCH_VERSION="$(uv run --no-sync python -c 'import torch; print(torch.__version__.split("+")[0])')"
uv pip install --reinstall-package torch --torch-backend "$TORCH_BACKEND" "torch==$TORCH_VERSION"
uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(), "no CUDA GPU"; print(torch.__version__, torch.cuda.get_device_name(0))'

uv run --no-sync edgeveil data download
uv run --no-sync edgeveil data prepare
uv run --no-sync edgeveil data augment
cp data/openpii/clean/report.json "$OUTPUT_DIR/prepare_report.json"
cp data/openpii/augmented/report.json "$OUTPUT_DIR/augment_report.json"

uv run --no-sync edgeveil train --preset distilbert --device-batch-size "$DEVICE_BATCH_SIZE" 2>&1 | tee "$OUTPUT_DIR/train.log"

tar -czf "$OUTPUT_DIR/distilbert-model.tar.gz" --exclude checkpoints -C models distilbert
cp models/distilbert/training_run.json "$OUTPUT_DIR/training_run.json"
