#!/usr/bin/env bash
set -euo pipefail

TRAIN_MANIFEST="${TRAIN_MANIFEST:-data/public/ipn/airdesk-train-ipn-all/tcn-v2-ipn-all-train-manifest.json}"
TEST_MANIFEST="${TEST_MANIFEST:-data/public/ipn/airdesk-test-ipn-all/tcn-v2-ipn-all-test-manifest.json}"
MODEL_OUT="${MODEL_OUT:-data/models/gestures/tcn-v2-ipn-all-80ep-h64-l4.pt}"
EVAL_OUT="${EVAL_OUT:-data/evaluations/ipn-all/tcn-v2-ipn-all-80ep-h64-l4-final-frame-heads.json}"
LOG_PATH="${LOG_PATH:-data/logs/tcn-v2-ipn-all-80ep-h64-l4-$(date +%Y%m%d-%H%M%S).log}"

EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-128}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-64}"
LEVELS="${LEVELS:-4}"
KERNEL_SIZE="${KERNEL_SIZE:-3}"
DROPOUT="${DROPOUT:-0.15}"
POSITIVE_WEIGHT_CAP="${POSITIVE_WEIGHT_CAP:-30}"
BOUNDARY_POSITIVE_WEIGHT_MULTIPLIER="${BOUNDARY_POSITIVE_WEIGHT_MULTIPLIER:-2.0}"
FOCAL_GAMMA="${FOCAL_GAMMA:-1.0}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.2}"
SEED="${SEED:-7}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"

mkdir -p "$(dirname "$MODEL_OUT")" "$(dirname "$EVAL_OUT")" "$(dirname "$LOG_PATH")"
exec > >(tee -a "$LOG_PATH") 2>&1

if [[ ! -f "$TRAIN_MANIFEST" ]]; then
  echo "missing train manifest: $TRAIN_MANIFEST" >&2
  exit 1
fi
if [[ ! -f "$TEST_MANIFEST" ]]; then
  echo "missing test manifest: $TEST_MANIFEST" >&2
  exit 1
fi

echo "AirDesk IPN-all TCN v2 training"
echo "started_at=$(date --iso-8601=seconds)"
echo "train_manifest=$TRAIN_MANIFEST"
echo "test_manifest=$TEST_MANIFEST"
echo "model_out=$MODEL_OUT"
echo "eval_out=$EVAL_OUT"
echo "log_path=$LOG_PATH"
echo "device=$DEVICE"

if [[ "$DEVICE" == "cuda" ]]; then
uv run python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False")

device_name = torch.cuda.get_device_name(0)
print(f"cuda_device={device_name}")
if "T550" not in device_name:
    raise SystemExit(f"Expected NVIDIA T550 Laptop GPU, got {device_name!r}")

props = torch.cuda.get_device_properties(0)
print(f"cuda_total_memory_mib={props.total_memory // (1024 * 1024)}")
sys.exit(0)
PY
fi

uv run airdesk gesture train-tcn-v2 \
  --manifest "$TRAIN_MANIFEST" \
  --out "$MODEL_OUT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --hidden-channels "$HIDDEN_CHANNELS" \
  --levels "$LEVELS" \
  --kernel-size "$KERNEL_SIZE" \
  --dropout "$DROPOUT" \
  --positive-weight-cap "$POSITIVE_WEIGHT_CAP" \
  --boundary-positive-weight-multiplier "$BOUNDARY_POSITIVE_WEIGHT_MULTIPLIER" \
  --focal-gamma "$FOCAL_GAMMA" \
  --validation-fraction "$VALIDATION_FRACTION" \
  --seed "$SEED" \
  --device "$DEVICE"

uv run airdesk gesture evaluate-tcn-v2-heads \
  --manifest "$TEST_MANIFEST" \
  --model "$MODEL_OUT" \
  --out "$EVAL_OUT" \
  --batch-size "$EVAL_BATCH_SIZE" \
  --device "$DEVICE"

echo "finished_at=$(date --iso-8601=seconds)"
