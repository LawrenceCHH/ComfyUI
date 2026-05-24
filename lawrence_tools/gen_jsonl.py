#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
# 預設內建 positive / negative list
# 你可以直接把這兩個 list 換成現在在用的那串長 prompt
'''
uv run gen_jsonl.py \
  --output prompts_style_sweep.jsonl \
  --run-name kazuma_style_sweep \
  --width 832 --height 1216 \
  --lora "Kazuma_Kiraboshi_-_Illustrious.safetensors" \
  --lora-strength-list 0.9 0.7 0.5 \
  --cfg-list 8.0 7.0 6.0 5.0 \
  --seed-range 42 42 \
  --steps 28 \
  --images-per-seed 1
'''
pos_list = [
    "kazuma kiraboshi, #Asuma #gay #Bara #muscle, muscular bara, dark skin, red eyes, black spiky hair, large pectorals, sweaty, locker room, completely naked, ahegao, heavy blushing, embarrassed, kneeling, full nelson, triple anal penetration, three thick cocks in one anus, single stretched anus, one anus stretched extremely wide, double penetration in one hole, triple penetration in same anus, extreme anal gape, loose prolapsing anus, anal stretching, bbc, thick veiny cocks, cum on face, hard dick, thrusting, surrounded by muscular men, gangbang, dynamic pose, masterpiece, score_9, score_8_up, detailed sweat",

    "kazuma kiraboshi, #gay #Bara #muscle, bara male, dark skin, red eyes, wrestling arena, naked, doggystyle, triple penetration, three cocks stretching one anus, single gaping anus being stretched by three dicks, extreme anal expansion, prolapse, loose anus, ahegao, tongue out, cum overflow, hard thrusting, masterpiece, score_9, score_8_up",

    "kazuma kiraboshi, #Asuma #gay #Bara #muscle, lying on back, mating press, triple anal, three thick cocks inside single anus, one extremely stretched anus, anal stretching to limit, gaping prolapsed anus, ahegao, blushing, bukkake, detailed male anatomy, masterpiece, score_9, best quality",
    "threesome, #Asuma #gay #Bara #muscle #AIart️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️️, black eyes, forehead protector, konohagakure symbol, stubble, beard, masterpiece,score_9, score_8_up, score_7_up, source_anime, easynegative, 1boy, solo, bandaid on face, bandaged hands, dark skin, black hair, spiked hair, red eyes, embarrassed, blushing, thong, ass focus muscular, toned body, bara, big pectorals, ((wrestling arena, stadium backgrounds)), sweaty, dripping sweat, sweat, sweatdrop, , hot, blush, muscular, toned male, athletic build, large pectorals, Naked, nude, locker room, full nelson, legs up, leaning back, , blushing, embarrassed, humiliated, led on a leash,, surrounded by penises, 6+boys, surrounded by penises, dynamic pose, cum on face, anal sex, dick in ass, gang bang, ahe gao face,ribs, pubic hair, spread anus,  anal stretch, loose anus, bbc, gapping, loose ass, prolapsing, anal sex, double penetration, rape, slit anus, squirting, kneel, hard dick, thrusting, prolapse"
]

neg_list = [
    "score_6, score_5, score_4, lowres, blurry, bad anatomy, bad proportions, deformed, extra anus, multiple anus, two anuses, three anuses, split anus, separate holes, multiple holes, extra holes, disconnected anus, pussy, vagina, female genitalia, breasts, female chest, (bad anus:1.3), deformed anus, extra penile, extra dicks, mutated penis, worst quality, low quality, realistic, 3d",
    
    "score_6, score_5, score_4, watercolor, lowres, low detail, text, blurry, simple background, low background, monochrome, monochrome background, blurry background, multiple_penises, bad anatomy, bad proportions, deformed, deformed anatomy, messy drawing, amateur drawing, ugly face, bad face, bad teeth, (interlocked fingers, badly drawn hands and fingers, anatomically incorrect hands, bad anatomy), deformed fingers, deformed arms, deformed hands, mutated hands, mutated fingers, glitch, deformed face, mutated tail, double tail, poorly drawn, bad art, boring, deformed, bad composition, crappy artwork, bad lighting, worst quality, poorly drawn, bad art, deformed, necklace, collar, realistic, 3d, facial hair, bald, long neck, hat on head,"
]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="輸出的 JSONL 檔案路徑")

    # 多組 positive / negative，改為選填
    ap.add_argument(
        "--base-prompts",
        nargs="+",
        required=False,
        help="一個或多個正向 prompt（pp0, pp1 ...）",
    )
    ap.add_argument(
        "--negatives",
        nargs="+",
        required=False,
        help="一個或多個 negative prompt（np0, np1 ...）",
    )

    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--steps", type=int, default=28)

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



    # 如果 CLI 有帶，就覆蓋預設 list；沒帶就用內建 list
    if args.base_prompts and len(args.base_prompts) > 0:
        base_prompts = args.base_prompts
    else:
        base_prompts = pos_list

    if args.negatives and len(args.negatives) > 0:
        negatives = args.negatives
    else:
        negatives = neg_list

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seed_start, seed_end = args.seed_range
    width = args.width
    height = args.height
    lora_name = args.lora
    steps = args.steps
    run_name = args.run_name

    with out_path.open("w", encoding="utf-8") as f:
        for pp_idx, base_prompt in enumerate(base_prompts):
            for np_idx, negative in enumerate(negatives):
                for lora_strength in args.lora_strength_list:
                    for cfg in args.cfg_list:
                        for seed in range(seed_start, seed_end + 1):
                            for image_index in range(1, args.images_per_seed + 1):
                                job_id = (
                                    f"kazu_cfg{cfg}_ls{lora_strength}_"
                                    f"s{seed}_img{image_index}_"
                                    f"pp{pp_idx}_np{np_idx}"
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
                                    "positive_index": pp_idx,
                                    "negative_index": np_idx,
                                }
                                f.write(json.dumps(row, ensure_ascii=False))
                                f.write("\n")

    print(f"wrote JSONL to {out_path}")


if __name__ == "__main__":
    main()