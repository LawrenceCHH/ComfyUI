#!/usr/bin/env bash
#
# Usage:
#   bash bash_pipeline.sh                  # 預設 profile (kazuma)
#   PROFILE=endeavor bash bash_pipeline.sh # 切換 profile
#   CONFIG=profiles/endeavor.yaml bash bash_pipeline.sh  # 直接指定 config 檔
#
# 所有實驗參數都寫在 profiles/<PROFILE>.yaml；改參數請編輯 YAML。
#
set -euo pipefail

PROFILE="${PROFILE:-kazuma}"
CONFIG="${CONFIG:-profiles/${PROFILE}.yaml}"
# 顯示用：從 CONFIG 反推 profile 名，這樣 CONFIG=... 跟 PROFILE=... 兩種寫法 log 一致
PROFILE_LABEL="$(basename "$CONFIG" .yaml)"

BASE_URL="${BASE_URL:-http://127.0.0.1:8188}"
COMFY_OUTPUT_ROOT="${COMFY_OUTPUT_ROOT:-/home/lawrencechh/Lprojects/comfyui/output}"
FINAL_OUTPUT_ROOT="${FINAL_OUTPUT_ROOT:-/mnt/d/comfy_runs}"

if [[ ! -f "$CONFIG" ]]; then
  echo "config 不存在: $CONFIG" >&2
  exit 1
fi

# 從 YAML 抓 workflow / output JSONL 路徑
WORKFLOW=$(uv run --quiet gen_jsonl.py --config "$CONFIG" --print-field workflow)
PROMPTS=$(uv run --quiet gen_jsonl.py --config "$CONFIG" --print-field output)

echo "=== profile: $PROFILE_LABEL ($CONFIG) ==="
echo "    workflow = $WORKFLOW"
echo "    prompts  = $PROMPTS"
echo

uv run gen_jsonl.py --config "$CONFIG" && \
uv run comfy_batch_generate_from_base.py \
  --base-url "$BASE_URL" \
  --base-workflow "$WORKFLOW" \
  --prompts "$PROMPTS" \
  --comfy-output-root "$COMFY_OUTPUT_ROOT" \
  --final-output-root "$FINAL_OUTPUT_ROOT"
