#!/usr/bin/env python3
"""
gen_jsonl.py

依照 CLI 參數產生 prompts.jsonl：
每行一個 job，包含：
id / prompt / negative / seed / steps / cfg / width / height / lora / lora_strength / image_index / run_name

LoRA 強度：決定「像 Kazuma 到什麼程度」。
CFG：決定「多聽 prompt、還是多放手讓模型自由發揮」。
Seed：固定就可重現（方便對比），放範圍就多樣化構圖。
Steps：畫面精細度（足夠即可，不是風格主軸）。

uv run gen_jsonl.py \
  --output prompts_style_sweep.jsonl \
  --run-name kazuma_style_sweep \
  --base-prompt "threesome, #Asuma #gay #Bara #muscle #AIart️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️, black eyes, forehead protector, konohagakure symbol, stubble, beard, masterpiece,score_9, score_8_up, score_7_up, source_anime, easynegative, 1boy, solo, bandaid on face, bandaged hands, dark skin, black hair, spiked hair, red eyes, embarrassed, blushing, thong, ass focus muscular, toned body, bara, big pectorals, ((wrestling arena, stadium backgrounds)), sweaty, dripping sweat, sweat, sweatdrop, , hot, blush, muscular, toned male, athletic build, large pectorals, Naked, nude, locker room, full nelson, legs up, leaning back, , blushing, embarrassed, humiliated, led on a leash,, surrounded by penises, 6+boys, surrounded by penises, dynamic pose, cum on face, anal sex, dick in ass, gang bang, ahe gao face,ribs, pubic hair, spread anus,  anal stretch, loose anus, bbc, gapping, loose ass, prolapsing, anal sex, double penetration, rape, slit anus, squirting, kneel, hard dick, thrusting, prolapse" \
  --negative "score_6, score_5, score_4, watercolor, lowres, low detail, text, blurry, simple background, low background, monochrome, monochrome background, blurry background, multiple_penises, bad anatomy, bad proportions, deformed, deformed anatomy, messy drawing, amateur drawing, ugly face, bad face, bad teeth, (interlocked fingers, badly drawn hands and fingers, anatomically incorrect hands, bad anatomy), deformed fingers, deformed arms, deformed hands, mutated hands, mutated fingers, glitch, deformed face, mutated tail, double tail, poorly drawn, bad art, boring, deformed, bad composition, crappy artwork, bad lighting, worst quality, poorly drawn, bad art, deformed, necklace, collar, realistic, 3d, facial hair, bald, long neck, hat on head," \
  --width 832 --height 1216 \
  --lora "Kazuma_Kiraboshi_-_Illustrious.safetensors" \
  --lora-strength-list 0.9 0.7 0.5 \
  --cfg-list 8.0 7.0 6.0 5.0 \
  --seed-range 42 42 \
  --steps 28 \
  --images-per-seed 1
"""

import argparse
import json
from pathlib import Path
from typing import List


def parse_float_list(values: List[str]) -> List[float]:
    return [float(v) for v in values]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="輸出的 JSONL 檔案路徑")
    ap.add_argument("--base-prompt", required=True)
    ap.add_argument("--negative", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--steps", type=int, default=28)

    # 新增：run 名稱（實驗名稱 / 子資料夾名等）
    ap.add_argument(
        "--run-name",
        type=str,
        default="default_run",
        help="這次實驗的名稱（會寫進 JSONL，方便用來組檔名或子資料夾）",
    )

    ap.add_argument(
        "--lora-strength-list",
        nargs="+",
        type=float,
        required=True,
        help="例如: 0.6 0.7 0.8",
    )
    ap.add_argument(
        "--cfg-list",
        nargs="+",
        type=float,
        required=True,
        help="例如: 6.0 6.5 7.0",
    )
    ap.add_argument(
        "--seed-range",
        nargs=2,
        type=int,
        metavar=("SEED_START", "SEED_END"),
        required=True,
        help="包含兩端，例如: 0 9",
    )
    ap.add_argument(
        "--images-per-seed",
        type=int,
        default=1,
        help="同一組參數下產生幾張圖（用 image_index 區分）",
    )

    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seed_start, seed_end = args.seed_range
    base_prompt = args.base_prompt
    negative = args.negative
    width = args.width
    height = args.height
    lora_name = args.lora
    steps = args.steps
    run_name = args.run_name

    with out_path.open("w", encoding="utf-8") as f:
        for lora_strength in args.lora_strength_list:
            for cfg in args.cfg_list:
                for seed in range(seed_start, seed_end + 1):
                    for image_index in range(1, args.images_per_seed + 1):
                        job_id = (
                            f"kazu_cfg{cfg}_ls{lora_strength}_"
                            f"s{seed}_img{image_index}"
                        )
                        row = {
                            "id": job_id,
                            "run_name": run_name,
                            "prompt": base_prompt,
                            "negative": negative,
                            "seed": seed,
                            "steps": steps,
                            "cfg": cfg,
                            "width": width,
                            "height": height,
                            "lora": lora_name,
                            "lora_strength": lora_strength,
                            "image_index": image_index,
                        }
                        f.write(json.dumps(row, ensure_ascii=False))
                        f.write("\n")

    print(f"wrote JSONL to {out_path}")


if __name__ == "__main__":
    main()