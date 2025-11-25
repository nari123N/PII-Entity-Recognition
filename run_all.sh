#!/usr/bin/env bash
set -euo pipefail

# --------
# Config
# --------
MODEL_NAME="${MODEL_NAME:-distilbert-base-uncased}"
OUT_DIR="${OUT_DIR:-out}"
TRAIN_FILE="${TRAIN_FILE:-data/train.jsonl}"
DEV_FILE="${DEV_FILE:-data/dev.jsonl}"
TEST_FILE="${TEST_FILE:-data/test.jsonl}"
STRESS_FILE="${STRESS_FILE:-data/stress.jsonl}"

BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-5}"
MAX_LENGTH="${MAX_LENGTH:-256}"

DEVICE="${DEVICE:-cpu}"
THREADS="${THREADS:-2}"

LATENCY_RUNS="${LATENCY_RUNS:-50}"

PYTHON="python3"     # <<--- HERE: use python3

echo "Using Python: $PYTHON"
echo

mkdir -p "$OUT_DIR"
export OMP_NUM_THREADS="$THREADS"

echo "=== 1) TRAINING ==="
$PYTHON src/train.py \
  --model_name "$MODEL_NAME" \
  --train "$TRAIN_FILE" \
  --dev "$DEV_FILE" \
  --out_dir "$OUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --max_length "$MAX_LENGTH" \
  --device "$DEVICE"

echo
echo "=== 2) PREDICT (DEV) ==="
DEV_PRED="$OUT_DIR/dev_pred.json"
$PYTHON src/predict.py \
  --model_dir "$OUT_DIR" \
  --input "$DEV_FILE" \
  --output "$DEV_PRED" \
  --max_length "$MAX_LENGTH" \
  --device "$DEVICE" \
  --threads "$THREADS" \
  --drop_pii_when_verify_fails

echo
echo "=== 3) EVALUATE (DEV) ==="
$PYTHON src/eval_span_f1.py --gold "$DEV_FILE" --pred "$DEV_PRED"

echo
echo "=== 4) PREDICT (STRESS) ==="
STRESS_PRED="$OUT_DIR/stress_pred.json"
$PYTHON src/predict.py \
  --model_dir "$OUT_DIR" \
  --input "$STRESS_FILE" \
  --output "$STRESS_PRED" \
  --max_length "$MAX_LENGTH" \
  --device "$DEVICE" \
  --threads "$THREADS" \
  --drop_pii_when_verify_fails

echo
echo "=== 5) EVALUATE (STRESS) ==="
$PYTHON src/eval_span_f1.py --gold "$STRESS_FILE" --pred "$STRESS_PRED"

echo
echo "=== 6) MEASURE LATENCY ==="
$PYTHON src/measure_latency.py \
  --model_dir "$OUT_DIR" \
  --input "$DEV_FILE" \
  --runs "$LATENCY_RUNS" \
  --device "$DEVICE"

echo
echo "=== ALL DONE ==="
