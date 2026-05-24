#!/usr/bin/env python3

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set

import requests

'''
uv run comfy_batch_generate_from_base.py \
  --base-url http://127.0.0.1:8188 \
  --base-workflow base_workflow.json \
  --prompts prompts_style_sweep.jsonl \
  --comfy-output-root /home/lawrencechh/Lprojects/comfyui/output \
  --final-output-root /mnt/d/comfy_runs
'''
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
    positive_index: int = 0
    negative_index: int = 0


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
                    positive_index=int(obj.get("positive_index", 0)),
                    negative_index=int(obj.get("negative_index", 0)),
                )
            )
    return jobs


def load_base_workflow(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt_from_base(
    base: Dict[str, Any],
    job: Job,
) -> Dict[str, Any]:
    wf = json.loads(json.dumps(base))

    wf["2"]["inputs"]["lora_name"] = job.lora
    wf["2"]["inputs"]["strength_model"] = job.lora_strength
    wf["3"]["inputs"]["seed"] = job.seed
    wf["3"]["inputs"]["steps"] = job.steps
    wf["3"]["inputs"]["cfg"] = job.cfg
    wf["6"]["inputs"]["width"] = job.width
    wf["6"]["inputs"]["height"] = job.height
    wf["7"]["inputs"]["text"] = job.prompt
    wf["8"]["inputs"]["text"] = job.negative

    return wf


def post_prompt(base_url: str, workflow: Dict[str, Any]) -> str:
    payload = {"prompt": workflow}
    resp = requests.post(f"{base_url.rstrip('/')}/prompt", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["prompt_id"]


def wait_for_done(base_url: str, prompt_id: str) -> Dict[str, Any]:
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
            return entry
        time.sleep(1)


def get_generated_images_from_history(history_entry: Dict[str, Any], comfy_output_root: Path) -> List[Path]:
    """從 ComfyUI 的歷史紀錄中萃取出實際生成的圖片檔案路徑"""
    generated_files = []
    outputs = history_entry.get("outputs", {})
    
    for node_id, node_data in outputs.items():
        if "images" in node_data:
            for img in node_data["images"]:
                subfolder = img.get("subfolder", "")
                filename = img.get("filename", "")
                
                # 組合出實際在磁碟上的路徑
                if subfolder:
                    file_path = comfy_output_root / subfolder / filename
                else:
                    file_path = comfy_output_root / filename
                    
                if file_path.exists():
                    generated_files.append(file_path)
                    
    return generated_files


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Use base_workflow.json + JSONL to batch queue ComfyUI jobs and move output using API history"
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8188")
    ap.add_argument("--base-workflow", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument(
        "--comfy-output-root",
        required=False,
        default="/home/lawrencechh/Lprojects/comfyui/output",
        help="ComfyUI 的 output 根目錄",
    )
    ap.add_argument(
        "--final-output-root",
        required=False,
        help="全部跑完後，要把影像搬移到哪個根目錄",
    )
    args = ap.parse_args()

    base_url = args.base_url
    base_workflow_path = Path(args.base_workflow)
    prompts_path = Path(args.prompts)
    comfy_output_root = Path(args.comfy_output_root)
    final_output_root = Path(args.final_output_root) if args.final_output_root else None

    base_workflow = load_base_workflow(base_workflow_path)
    jobs = load_jobs(prompts_path)

    print(f"Loaded base workflow from {base_workflow_path}")
    print(f"Loaded {len(jobs)} jobs from {prompts_path}")

    for job in jobs:
        print(f"\n=== Running job: {job.id} ===")
        wf = build_prompt_from_base(base_workflow, job)
        
        prompt_id = post_prompt(base_url, wf)
        print(f"  queued prompt_id={prompt_id}, waiting to finish...")
        
        # 取得完成後的歷史紀錄
        history_entry = wait_for_done(base_url, prompt_id)
        print(f"  job {job.id} done (Comfy history status=completed)")

        # 透過 API 回傳的紀錄精準抓出這次生成的圖片
        image_files = get_generated_images_from_history(history_entry, comfy_output_root)

        if not image_files:
            print("  [錯誤] 任務完成，但從 API 紀錄中沒有找到任何生成的圖片！(可能是快取略過或無 SaveImage 節點)")
            continue
        
        print(f"  API 確認生成了 {len(image_files)} 個檔案。")

        src_dirs_to_clean: Set[Path] = set()

        for idx, file_path in enumerate(image_files):
            print(f"    [原始影像] {file_path}")
            src_dirs_to_clean.add(file_path.parent)

            # 組合新檔名: RUN_NAME_loras0.9_cfg7.0_seed42_img1
            new_stem = f"{job.run_name}_loras{job.lora_strength}_cfg{job.cfg}_seed{job.seed}_img{job.image_index}"
            
            if len(image_files) > 1 and idx > 0:
                new_stem += f"_{idx+1}"

            # 決定目標資料夾 (以 run_name 為準)
            if final_output_root is not None:
                dst_dir = final_output_root / job.run_name
            else:
                dst_dir = comfy_output_root / job.run_name
            
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_file = dst_dir / f"{new_stem}{file_path.suffix}"
            
            # 處理檔名重複 (避免覆寫)
            counter = 1
            while dst_file.exists():
                dst_file = dst_dir / f"{new_stem}_{counter}{file_path.suffix}"
                counter += 1

            # 搬移檔案
            shutil.move(str(file_path), str(dst_file))
            print(f"    -> [重新命名並搬移] {dst_file}")

        # 清理可能因為搬移檔案而變空的原始子資料夾 (避開 output 根目錄)
        for src_dir in src_dirs_to_clean:
            if src_dir.exists() and src_dir != comfy_output_root and not any(src_dir.iterdir()):
                try:
                    src_dir.rmdir()
                    print(f"    -> [清理] 原始資料夾已清空並刪除: {src_dir}")
                except OSError:
                    pass

    print("\nAll jobs done.")


if __name__ == "__main__":
    main()
