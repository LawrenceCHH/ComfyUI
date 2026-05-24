#!/usr/bin/env python3

import argparse
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

'''
uv run comfy_batch_generate_from_base.py \
  --base-url http://127.0.0.1:8188 \
  --base-workflow base_workflow.json \
  --prompts prompts_style_sweep.jsonl \
  --comfy-output-root /home/lawrencechh/Lprojects/comfyui/output \
  --final-output-root /mnt/d/comfy_runs
'''


LORA_LOADER_CLASS_TYPES = {"LoraLoaderModelOnly", "LoraLoader"}


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
    loras: List[Dict[str, Any]] = field(default_factory=list)
    image_index: int = 1
    positive_index: int = 0
    negative_index: int = 0
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None


def load_jobs(jsonl_path: Path) -> List[Job]:
    jobs: List[Job] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # LoRA 支援新舊兩種格式：
            #   新：loras = [{"name": str, "strength": float}, ...]
            #   舊：lora = str, lora_strength = float
            if "loras" in obj and obj["loras"]:
                loras = [
                    {"name": str(l["name"]), "strength": float(l["strength"])}
                    for l in obj["loras"]
                ]
            elif "lora" in obj:
                loras = [
                    {"name": str(obj["lora"]), "strength": float(obj.get("lora_strength", 1.0))}
                ]
            else:
                loras = []

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
                    loras=loras,
                    image_index=int(obj.get("image_index", 1)),
                    positive_index=int(obj.get("positive_index", 0)),
                    negative_index=int(obj.get("negative_index", 0)),
                    sampler_name=obj.get("sampler_name"),
                    scheduler=obj.get("scheduler"),
                )
            )
    return jobs


def load_base_workflow(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_nodes_by_class(wf: Dict[str, Any], class_type: str) -> List[Tuple[str, Dict[str, Any]]]:
    return [(nid, n) for nid, n in wf.items() if isinstance(n, dict) and n.get("class_type") == class_type]


def _resolve_input_link(wf: Dict[str, Any], node: Dict[str, Any], input_name: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """從 node.inputs[input_name] 抓 [target_node_id, slot] 連線，回傳 (target_id, target_node)。
    若該欄位是純值（非連線），回傳 (None, None)。"""
    ref = node.get("inputs", {}).get(input_name)
    if isinstance(ref, list) and len(ref) >= 1:
        target_id = str(ref[0])
        return target_id, wf.get(target_id)
    return None, None


@dataclass
class WorkflowSlots:
    ksampler_id: str
    positive_id: Optional[str]
    negative_id: Optional[str]
    latent_id: Optional[str]
    lora_ids: List[str]  # 依資料流順序：index 0 = 最靠近 Checkpoint 的 LoRA


def locate_workflow_slots(wf: Dict[str, Any]) -> WorkflowSlots:
    """依 class_type + 連線追溯，動態找出 workflow 中要改的節點。"""
    ksamplers = _find_nodes_by_class(wf, "KSampler")
    if not ksamplers:
        raise RuntimeError("workflow 中找不到 KSampler 節點")
    if len(ksamplers) > 1:
        print(f"  [警告] workflow 中有 {len(ksamplers)} 個 KSampler，只覆寫第一個 ({ksamplers[0][0]})")
    ksampler_id, ksampler = ksamplers[0]

    positive_id, _ = _resolve_input_link(wf, ksampler, "positive")
    negative_id, _ = _resolve_input_link(wf, ksampler, "negative")
    latent_id, _ = _resolve_input_link(wf, ksampler, "latent_image")

    # LoRA chain：從 KSampler.model 一路往回追，直到不是 LoraLoader
    lora_chain: List[str] = []
    cur_id, cur_node = _resolve_input_link(wf, ksampler, "model")
    seen: Set[str] = set()
    while cur_node and cur_node.get("class_type") in LORA_LOADER_CLASS_TYPES:
        if cur_id in seen:
            break  # 防止迴圈
        seen.add(cur_id)
        lora_chain.append(cur_id)
        cur_id, cur_node = _resolve_input_link(wf, cur_node, "model")

    # 反轉為「資料流順序」：index 0 = 最靠近 Checkpoint 的 LoRA
    lora_chain.reverse()

    return WorkflowSlots(
        ksampler_id=ksampler_id,
        positive_id=positive_id,
        negative_id=negative_id,
        latent_id=latent_id,
        lora_ids=lora_chain,
    )


def build_prompt_from_base(
    base: Dict[str, Any],
    job: Job,
    slots: WorkflowSlots,
) -> Dict[str, Any]:
    wf = json.loads(json.dumps(base))

    ks_inputs = wf[slots.ksampler_id]["inputs"]
    ks_inputs["seed"] = job.seed
    ks_inputs["steps"] = job.steps
    ks_inputs["cfg"] = job.cfg
    if job.sampler_name:
        ks_inputs["sampler_name"] = job.sampler_name
    if job.scheduler:
        ks_inputs["scheduler"] = job.scheduler

    if slots.latent_id is not None:
        lat_inputs = wf[slots.latent_id]["inputs"]
        lat_inputs["width"] = job.width
        lat_inputs["height"] = job.height

    if slots.positive_id is not None:
        wf[slots.positive_id]["inputs"]["text"] = job.prompt
    if slots.negative_id is not None:
        wf[slots.negative_id]["inputs"]["text"] = job.negative

    if job.loras:
        if len(job.loras) > len(slots.lora_ids):
            print(
                f"  [警告] JSONL 指定 {len(job.loras)} 個 LoRA，但 workflow 只有 "
                f"{len(slots.lora_ids)} 個 LoRA loader；多餘的會被忽略"
            )
        for i, lora_spec in enumerate(job.loras[: len(slots.lora_ids)]):
            lora_inputs = wf[slots.lora_ids[i]]["inputs"]
            lora_inputs["lora_name"] = lora_spec["name"]
            lora_inputs["strength_model"] = lora_spec["strength"]
            if "strength_clip" in lora_inputs:
                lora_inputs["strength_clip"] = lora_spec["strength"]

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

                if subfolder:
                    file_path = comfy_output_root / subfolder / filename
                else:
                    file_path = comfy_output_root / filename

                if file_path.exists():
                    generated_files.append(file_path)

    return generated_files


def _format_lora_strengths(loras: List[Dict[str, Any]]) -> str:
    if not loras:
        return "none"
    return "-".join(str(l["strength"]) for l in loras)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Use a ComfyUI API-format workflow JSON + JSONL to batch queue jobs and move output. "
                    "Workflow node slots are located dynamically via class_type, so any compatible workflow works."
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
    slots = locate_workflow_slots(base_workflow)

    print(f"Loaded base workflow from {base_workflow_path}")
    print(
        f"Workflow slots: KSampler={slots.ksampler_id}, "
        f"positive={slots.positive_id}, negative={slots.negative_id}, "
        f"latent={slots.latent_id}, lora_chain={slots.lora_ids}"
    )
    print(f"Loaded {len(jobs)} jobs from {prompts_path}")

    for job in jobs:
        print(f"\n=== Running job: {job.id} ===")
        wf = build_prompt_from_base(base_workflow, job, slots)

        prompt_id = post_prompt(base_url, wf)
        print(f"  queued prompt_id={prompt_id}, waiting to finish...")

        history_entry = wait_for_done(base_url, prompt_id)
        print(f"  job {job.id} done (Comfy history status=completed)")

        image_files = get_generated_images_from_history(history_entry, comfy_output_root)

        if not image_files:
            print("  [錯誤] 任務完成，但從 API 紀錄中沒有找到任何生成的圖片！(可能是快取略過或無 SaveImage 節點)")
            continue

        print(f"  API 確認生成了 {len(image_files)} 個檔案。")

        src_dirs_to_clean: Set[Path] = set()

        for idx, file_path in enumerate(image_files):
            print(f"    [原始影像] {file_path}")
            src_dirs_to_clean.add(file_path.parent)

            # 組合新檔名: RUN_NAME_loras0.7-1.0_cfg7.0_seed42_img1_pp0_np0
            lora_tag = _format_lora_strengths(job.loras)
            new_stem = (
                f"{job.run_name}_loras{lora_tag}_cfg{job.cfg}"
                f"_seed{job.seed}_img{job.image_index}"
                f"_pp{job.positive_index}_np{job.negative_index}"
            )

            if len(image_files) > 1 and idx > 0:
                new_stem += f"_{idx+1}"

            if final_output_root is not None:
                dst_dir = final_output_root / job.run_name
            else:
                dst_dir = comfy_output_root / job.run_name

            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_file = dst_dir / f"{new_stem}{file_path.suffix}"

            counter = 1
            while dst_file.exists():
                dst_file = dst_dir / f"{new_stem}_{counter}{file_path.suffix}"
                counter += 1

            shutil.move(str(file_path), str(dst_file))
            print(f"    -> [重新命名並搬移] {dst_file}")

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
