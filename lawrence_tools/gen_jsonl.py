#!/usr/bin/env python3
"""
從 YAML profile (或 CLI 參數) 產生 JSONL，餵給 comfy_batch_generate_from_base.py。

CLI > YAML > 內建預設。沒有 --config 也可以全用 CLI 跑。

範例:
  uv run gen_jsonl.py --config profiles/endeavor.yaml
  uv run gen_jsonl.py --config profiles/kazuma.yaml --seed-range 100 105  # CLI 覆寫
  uv run gen_jsonl.py --config profiles/endeavor.yaml --print-field workflow

純 CLI 用法 (向後相容):
  uv run gen_jsonl.py \
    --output prompts.jsonl --run-name foo \
    --width 832 --height 1216 \
    --lora "X.safetensors" --lora-strength-list 0.7 \
    --cfg-list 6.0 --seed-range 42 45 --steps 28 \
    --base-prompts "prompt A" "prompt B" \
    --negatives "neg A"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _parse_lora_spec(spec: str) -> Tuple[str, float]:
    """CLI 用：解析 "name:strength" 或 "name=strength"，沒給 strength 預設 1.0"""
    for sep in (":", "="):
        if sep in spec:
            name, strength_str = spec.rsplit(sep, 1)
            return name.strip(), float(strength_str.strip())
    return spec.strip(), 1.0


def _normalize_lora_entry(entry: Any) -> Dict[str, Any]:
    """YAML 中 LoRA 可寫成 {name, strength} dict 或 'name:strength' 字串。"""
    if isinstance(entry, dict):
        return {"name": str(entry["name"]), "strength": float(entry.get("strength", 1.0))}
    if isinstance(entry, str):
        name, strength = _parse_lora_spec(entry)
        return {"name": name, "strength": strength}
    raise ValueError(f"無法解析的 LoRA 項目: {entry!r}")


def _format_lora_tag(loras: List[Dict[str, Any]]) -> str:
    return "-".join(str(l["strength"]) for l in loras) if loras else "none"


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"config 檔內容不是 dict: {path}")
    return cfg


def _cfg_get(cfg: Dict[str, Any], *keys, default=None):
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _resolve_lora_combos(args: argparse.Namespace, cfg: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """
    回傳 list of combos；每個 combo 是 list of {name, strength}。
    決定優先序：CLI --loras > CLI --lora+--lora-strength-list > cfg lora_combos > cfg loras
    """
    if args.loras:
        if args.lora or args.lora_strength_list:
            raise SystemExit("--loras 與 --lora/--lora-strength-list 互斥")
        combo = [
            {"name": name, "strength": strength}
            for name, strength in (_parse_lora_spec(s) for s in args.loras)
        ]
        return [combo]

    if args.lora and args.lora_strength_list:
        return [[{"name": args.lora, "strength": s}] for s in args.lora_strength_list]
    if args.lora or args.lora_strength_list:
        raise SystemExit("--lora 與 --lora-strength-list 要一起給")

    cfg_combos = _cfg_get(cfg, "lora_combos")
    if cfg_combos:
        return [[_normalize_lora_entry(l) for l in combo] for combo in cfg_combos]

    cfg_loras = _cfg_get(cfg, "loras")
    if cfg_loras:
        return [[_normalize_lora_entry(l) for l in cfg_loras]]

    raise SystemExit(
        "找不到 LoRA 設定：請在 config 提供 loras / lora_combos，或 CLI 給 --loras 或 --lora+--lora-strength-list"
    )


def _require(value, name: str):
    if value is None:
        raise SystemExit(f"缺少必要欄位：{name} (請在 config 或 CLI 提供)")
    return value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML profile 路徑")
    ap.add_argument("--print-field", help="只印出 config 中某個 top-level key 的值後結束（bash_pipeline 用）")

    ap.add_argument("--output", help="輸出的 JSONL 路徑（也可寫在 config 的 output）")
    ap.add_argument("--run-name", help="實驗名稱（會寫進 JSONL）")

    ap.add_argument("--base-prompts", nargs="+", help="正向 prompt list；覆寫 config 的 prompts.positive")
    ap.add_argument("--negatives", nargs="+", help="負向 prompt list；覆寫 config 的 prompts.negative")

    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--steps", type=int)

    # LoRA: 單 LoRA + sweep (舊) 或多 LoRA (新)
    ap.add_argument("--lora", help="[單 LoRA 模式] LoRA 檔名；配 --lora-strength-list")
    ap.add_argument("--lora-strength-list", nargs="+", type=float)
    ap.add_argument("--loras", nargs="+", help="[多 LoRA] 例: 'A.safetensors:1.0' 'B.safetensors:0.8'")

    ap.add_argument("--cfg-list", nargs="+", type=float)
    ap.add_argument("--seed-range", nargs=2, type=int, metavar=("SEED_START", "SEED_END"))
    ap.add_argument("--images-per-seed", type=int)

    ap.add_argument("--sampler-list", nargs="+", help="(選填) sampler_name sweep")
    ap.add_argument("--scheduler-list", nargs="+", help="(選填) scheduler sweep")

    args = ap.parse_args()

    cfg = _load_config(args.config)

    # 給 bash_pipeline 抓 workflow / output 路徑用
    if args.print_field:
        val = cfg.get(args.print_field)
        if val is None:
            print(f"[gen_jsonl] config 中沒有 '{args.print_field}'", file=sys.stderr)
            sys.exit(1)
        print(val)
        return

    # === 解析每個欄位：CLI > config > error/default ===
    output = _require(args.output or cfg.get("output"), "output")
    run_name = args.run_name or cfg.get("run_name") or "default_run"

    width = _require(args.width or _cfg_get(cfg, "image", "width"), "width")
    height = _require(args.height or _cfg_get(cfg, "image", "height"), "height")
    steps = args.steps or _cfg_get(cfg, "sampling", "steps") or 28

    cfg_list = args.cfg_list or _cfg_get(cfg, "sampling", "cfg_list")
    cfg_list = _require(cfg_list, "cfg_list")

    seed_range = args.seed_range or _cfg_get(cfg, "sampling", "seed_range")
    seed_range = _require(seed_range, "seed_range")
    if len(seed_range) != 2:
        raise SystemExit("seed_range 必須是 [start, end] 兩個整數")
    seed_start, seed_end = int(seed_range[0]), int(seed_range[1])

    images_per_seed = args.images_per_seed or _cfg_get(cfg, "sampling", "images_per_seed") or 1

    sampler_list = args.sampler_list or _cfg_get(cfg, "sampling", "sampler_list") or [None]
    scheduler_list = args.scheduler_list or _cfg_get(cfg, "sampling", "scheduler_list") or [None]

    base_prompts = args.base_prompts or _cfg_get(cfg, "prompts", "positive")
    base_prompts = _require(base_prompts, "prompts.positive / --base-prompts")
    negatives = args.negatives or _cfg_get(cfg, "prompts", "negative")
    negatives = _require(negatives, "prompts.negative / --negatives")

    lora_combos = _resolve_lora_combos(args, cfg)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for pp_idx, base_prompt in enumerate(base_prompts):
            for np_idx, negative in enumerate(negatives):
                for loras in lora_combos:
                    for cfg_val in cfg_list:
                        for sampler in sampler_list:
                            for scheduler in scheduler_list:
                                for seed in range(seed_start, seed_end + 1):
                                    for image_index in range(1, images_per_seed + 1):
                                        lora_tag = _format_lora_tag(loras)
                                        sampler_tag = f"_smp{sampler}" if sampler else ""
                                        sched_tag = f"_sch{scheduler}" if scheduler else ""
                                        job_id = (
                                            f"{run_name}_cfg{cfg_val}_ls{lora_tag}"
                                            f"{sampler_tag}{sched_tag}"
                                            f"_s{seed}_img{image_index}"
                                            f"_pp{pp_idx}_np{np_idx}"
                                        )
                                        row = {
                                            "id": job_id,
                                            "run_name": run_name,
                                            "prompt": base_prompt,
                                            "negative": negative,
                                            "seed": seed,
                                            "steps": steps,
                                            "cfg": cfg_val,
                                            "width": width,
                                            "height": height,
                                            "loras": loras,
                                            "image_index": image_index,
                                            "positive_index": pp_idx,
                                            "negative_index": np_idx,
                                        }
                                        if sampler:
                                            row["sampler_name"] = sampler
                                        if scheduler:
                                            row["scheduler"] = scheduler
                                        f.write(json.dumps(row, ensure_ascii=False))
                                        f.write("\n")
                                        written += 1

    print(f"wrote {written} jobs to {out_path}")


if __name__ == "__main__":
    main()
