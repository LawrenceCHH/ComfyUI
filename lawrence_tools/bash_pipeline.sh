uv run gen_jsonl.py \
  --output prompts_style_sweep.jsonl \
  --run-name kazuma_style_sweep \
  --width 896 --height 1344 \
  --lora "Kazuma_Kiraboshi_-_Illustrious.safetensors" \
  --lora-strength-list 0.7 \
  --cfg-list 6.0 \
  --seed-range 42 45 \
  --steps 28 \
  --images-per-seed 1 && \
uv run comfy_batch_generate_from_base.py \
  --base-url http://127.0.0.1:8188 \
  --base-workflow base_workflow.json \
  --prompts prompts_style_sweep.jsonl \
  --comfy-output-root /home/lawrencechh/Lprojects/comfyui/output \
  --final-output-root /mnt/d/comfy_runs