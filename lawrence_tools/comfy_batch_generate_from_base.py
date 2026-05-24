#!/usr/bin/env python3

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

"""
使用方式範例：

uv run comfy_batch_generate_from_base.py \
  --base-url http://127.0.0.1:8188 \
  --base-workflow base_workflow.json \
  --prompts prompts_style_sweep.jsonl \
  --output-subdir kazuma_style_sweep \
  --comfy-output-root /home/lawrencechh/Lprojects/comfyui/output \
  --final-output-root /mnt/d/comfy_runs

流程：
1. ComfyUI 用 SaveImageExtended 把圖存到：
   /home/.../comfyui/output/kazuma_style_sweep
2. 全部 job 跑完後，把這個資料夾搬到：
   /mnt/bigdisk/comfy_runs/kazuma_style_sweep
"""


@dataclass
class Job:
    id: str
    run_name: str
    prompt: str
    negative: str
    seed: int
    steps: int
    cfg: float
    width: int
    height: int
    lora: str
    lora_strength: float
    image_index: int


def load_jobs(jsonl_path: Path) -> List[Job]:
    jobs: List[Job] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            jobs.append(
                Job(
                    id=obj["id"],
                    run_name=obj.get("run_name", "default_run"),
                    prompt=obj["prompt"],
                    negative=obj["negative"],
                    seed=int(obj["seed"]),
                    steps=int(obj["steps"]),
                    cfg=float(obj["cfg"]),
                    width=int(obj["width"]),
                    height=int(obj["height"]),
                    lora=obj["lora"],
                    lora_strength=float(obj["lora_strength"]),
                    image_index=int(obj.get("image_index", 1)),
                )
            )
    return jobs


def load_base_workflow(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt_from_base(
    base: Dict[str, Any],
    job: Job,
    output_subdir: Optional[str] = None,
) -> Dict[str, Any]:
    wf = json.loads(json.dumps(base))  # 深拷貝

    # 2: LoraLoaderModelOnly
    wf["2"]["inputs"]["lora_name"] = job.lora
    wf["2"]["inputs"]["strength_model"] = job.lora_strength

    # 3: KSampler
    wf["3"]["inputs"]["seed"] = job.seed
    wf["3"]["inputs"]["steps"] = job.steps
    wf["3"]["inputs"]["cfg"] = job.cfg

    # 6: EmptyLatentImage
    wf["6"]["inputs"]["width"] = job.width
    wf["6"]["inputs"]["height"] = job.height

    # 7: CLIPTextEncode (Prompt)
    wf["7"]["inputs"]["text"] = job.prompt

    # 8: CLIPTextEncode (Negative)
    wf["8"]["inputs"]["text"] = job.negative

    # 10: SaveImageExtended
    save_inputs = wf["10"]["inputs"]

    # 資料夾名稱：固定成 output_subdir（或 run_name），不再帶 euler_karras
    folder_prefix = output_subdir if output_subdir is not None else job.run_name

    filename_prefix = (
        f"{job.run_name}"
        f"_cfg{job.cfg}"
        f"_ls{job.lora_strength}"
        f"_seed{job.seed}"
        f"_img{job.image_index}"
    )

    save_inputs["foldername_prefix"] = folder_prefix
    save_inputs["filename_prefix"] = filename_prefix

    # 關鍵：清空 foldername_keys，避免自動加 _euler_karras
    save_inputs["foldername_keys"] = ""
    # 如果不想檔名再加 steps, cfg，也可以一起清空
    # save_inputs["filename_keys"] = ""

    return wf


def post_prompt(base_url: str, workflow: Dict[str, Any]) -> str:
    payload = {"prompt": workflow}
    resp = requests.post(f"{base_url.rstrip('/')}/prompt", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["prompt_id"]


def wait_for_done(base_url: str, prompt_id: str) -> None:
    while True:
        resp = requests.get(f"{base_url.rstrip('/')}/history/{prompt_id}")
        if resp.status_code == 404:
            time.sleep(1)
            continue
        resp.raise_for_status()
        data = resp.json()
        entry = data.get(prompt_id)
        if not entry:
            time.sleep(1)
            continue
        if entry.get("status", {}).get("completed"):
            break
        time.sleep(1)


def move_folder_after_run(
    comfy_output_root: Path,
    output_subdir: str,
    final_output_root: Path,
) -> None:
    src = comfy_output_root / output_subdir
    dst = final_output_root / output_subdir

    if not src.exists():
        print(f"[WARN] source folder not found: {src}")
        return

    final_output_root.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        backup = dst.with_name(dst.name + "_old")
        print(f"[INFO] {dst} exists, renaming to {backup}")
        dst.rename(backup)

    print(f"[INFO] moving folder {src} -> {dst}")
    shutil.move(str(src), str(dst))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Use base_workflow.json + JSONL to batch queue ComfyUI jobs and move output folder"
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8188")
    ap.add_argument("--base-workflow", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument(
        "--output-subdir",
        required=False,
        help="相對於 ComfyUI output 根目錄的子資料夾，例如 kazuma_style_sweep",
    )
    ap.add_argument(
        "--comfy-output-root",
        required=False,
        default="/home/lawrencechh/Lprojects/comfyui/output",
    )
    ap.add_argument(
        "--final-output-root",
        required=False,
        help="全部跑完後，要把 output-subdir 搬到哪個根目錄（任意位置）",
    )
    args = ap.parse_args()

    base_url = args.base_url
    base_workflow_path = Path(args.base_workflow)
    prompts_path = Path(args.prompts)
    output_subdir = args.output_subdir
    comfy_output_root = Path(args.comfy_output_root)
    final_output_root = Path(args.final_output_root) if args.final_output_root else None

    base_workflow = load_base_workflow(base_workflow_path)
    jobs = load_jobs(prompts_path)

    print(f"Loaded base workflow from {base_workflow_path}")
    print(f"Loaded {len(jobs)} jobs from {prompts_path}")
    if output_subdir:
        print(f"Comfy output subfolder: {output_subdir}")

    for job in jobs:
        print(f"\n=== Running job: {job.id} ===")
        wf = build_prompt_from_base(base_workflow, job, output_subdir=output_subdir)
        prompt_id = post_prompt(base_url, wf)
        print(f"  queued prompt_id={prompt_id}, waiting to finish...")
        wait_for_done(base_url, prompt_id)
        print(f"  job {job.id} done (images saved by SaveImageExtended node)")

    if output_subdir and final_output_root is not None:
        move_folder_after_run(comfy_output_root, output_subdir, final_output_root)

    print("\nAll jobs done.")


if __name__ == "__main__":
    main()