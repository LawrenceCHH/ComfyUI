#!/usr/bin/env python3
"""
Civitai Model Downloader
自動查詢 LoRA metadata、對應 Base Model 並下載
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────
# 設定區
# ─────────────────────────────────────────────
BASE_MODEL_IDS = {
    "SD 1.5":       None,   # 從 HuggingFace 下載，不走 Civitai
    "SDXL 1.0":     None,   # 同上
    "Pony":         "257749",
    "Illustrious":  "795765",
    "FLUX.1 D":     None,   # 從 HuggingFace 下載
    "FLUX.1 S":     None,
}

OUTPUT_DIRS = {
    "Checkpoint":  "models/checkpoints",
    "LORA":        "models/loras",
    "TextualInversion": "models/embeddings",
    "VAE":         "models/vae",
    "ControlNet":  "models/controlnet",
}

CIVITAI_API   = "https://civitai.com/api/v1"
CIVITAI_DL    = "https://civitai.com/api/download/models"
CHUNK_SIZE    = 8192


# ─────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────
def get_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def fetch_model_version(version_id: str, token: str) -> dict:
    url = f"{CIVITAI_API}/model-versions/{version_id}"
    resp = requests.get(url, headers=get_headers(token))
    resp.raise_for_status()
    return resp.json()


def fetch_model_info(model_id: str, token: str) -> dict:
    url = f"{CIVITAI_API}/models/{model_id}"
    resp = requests.get(url, headers=get_headers(token))
    resp.raise_for_status()
    return resp.json()


def download_file(url: str, dest: Path, token: str, filename: str = None):
    headers = get_headers(token)
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()

        # 從 Content-Disposition 取得原始檔名
        if filename is None:
            cd = r.headers.get("Content-Disposition", "")
            if "filename=" in cd:
                filename = cd.split("filename=")[-1].strip().strip('"')
            else:
                filename = url.split("/")[-1].split("?")[0]

        dest_path = dest / filename
        total = int(r.headers.get("Content-Length", 0))

        print(f"  → 儲存至：{dest_path}  ({total / 1024**3:.2f} GB)" if total else f"  → 儲存至：{dest_path}")

        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc=filename[:40], leave=True
        ) as bar:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                bar.update(len(chunk))

    return dest_path


# ─────────────────────────────────────────────
# 核心流程
# ─────────────────────────────────────────────
def download_model(version_id: str, token: str, force_type: str = None):
    print(f"\n[1/3] 查詢 model-version metadata（ID: {version_id}）...")
    meta = fetch_model_version(version_id, token)

    model_type  = force_type or meta.get("model", {}).get("type", "LORA")
    base_model  = meta.get("baseModel", "Unknown")
    files       = meta.get("files", [])
    model_name  = meta.get("model", {}).get("name", "unknown")

    if not files:
        print("  ✗ 無可下載的檔案")
        return

    # 選擇 safetensors，優先 pruned+fp16
    target_file = None
    for f in files:
        meta_f = f.get("metadata", {})
        if (f.get("type") == "Model"
                and f.get("name", "").endswith(".safetensors")
                and meta_f.get("fp") == "fp16"):
            target_file = f
            break
    if target_file is None:
        target_file = files[0]   # fallback

    print(f"  模型名稱：{model_name}")
    print(f"  類型：    {model_type}")
    print(f"  Base Model：{base_model}")
    print(f"  檔案：    {target_file['name']}  ({target_file.get('sizeKB', 0)/1024:.0f} MB)")

    # 決定輸出目錄
    out_dir_key = model_type if model_type in OUTPUT_DIRS else "LORA"
    out_dir = Path(OUTPUT_DIRS[out_dir_key])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 下載主模型
    print(f"\n[2/3] 下載模型...")
    dl_url = f"{CIVITAI_DL}/{version_id}?type=Model&format=SafeTensor"
    if target_file.get("metadata", {}).get("fp") == "fp16":
        dl_url += "&fp=fp16"
    if target_file.get("metadata", {}).get("size") == "pruned":
        dl_url += "&size=pruned"

    saved = download_file(dl_url, out_dir, token, target_file["name"])
    print(f"  ✓ 下載完成：{saved}")

    # 提示 Base Model 資訊
    print(f"\n[3/3] Base Model 對應資訊...")
    if base_model in BASE_MODEL_IDS and BASE_MODEL_IDS[base_model]:
        base_id = BASE_MODEL_IDS[base_model]
        print(f"  Base Model 為 [{base_model}]，Civitai ID: {base_id}")
        ans = input(f"  是否一併下載 Base Model？(y/N): ").strip().lower()
        if ans == "y":
            download_model(base_id, token, force_type="Checkpoint")
    elif base_model in ("SD 1.5", "SDXL 1.0", "FLUX.1 D", "FLUX.1 S"):
        hf_map = {
            "SD 1.5":   "runwayml/stable-diffusion-v1-5",
            "SDXL 1.0": "stabilityai/stable-diffusion-xl-base-1.0",
            "FLUX.1 D": "black-forest-labs/FLUX.1-dev",
            "FLUX.1 S": "black-forest-labs/FLUX.1-schnell",
        }
        print(f"  Base Model 為 [{base_model}]")
        print(f"  請從 HuggingFace 下載：https://huggingface.co/{hf_map[base_model]}")
        print(f"  建議指令：")
        print(f"    huggingface-cli download {hf_map[base_model]} --local-dir models/checkpoints/")
    else:
        print(f"  Base Model：{base_model}（請手動確認是否已下載）")

    return saved


# ─────────────────────────────────────────────
# CLI 介面
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Civitai 模型下載工具（自動對應 Base Model）",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "version_ids",
        nargs="+",
        help="一或多個 Civitai model-version ID（從模型頁 URL 取得）"
    )
    parser.add_argument(
        "--token", "-t",
        default=os.environ.get("CIVITAI_TOKEN", ""),
        help="Civitai API Token（預設讀取環境變數 CIVITAI_TOKEN）"
    )
    args = parser.parse_args()

    if not args.token:
        print("✗ 未提供 API Token，請設定環境變數 CIVITAI_TOKEN 或使用 --token 參數")
        sys.exit(1)

    for vid in args.version_ids:
        try:
            download_model(vid, args.token)
        except requests.HTTPError as e:
            print(f"  ✗ HTTP 錯誤 {e.response.status_code}：{e}")
        except KeyboardInterrupt:
            print("\n已中止下載")
            sys.exit(0)
        except Exception as e:
            print(f"  ✗ 錯誤：{e}")


if __name__ == "__main__":
    main()
