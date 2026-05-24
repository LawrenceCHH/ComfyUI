# ComfyUI 批次產圖完整教學（Civitai → ComfyUI → YAML Profile → 批次 runner）

這份文件補完整了 **環境安裝**、**workflow API JSON**、**YAML profile 系統**、**動態節點偵測**、**檔案搬移與重新命名** 與 **除錯建議**。整套流程以 `bash_pipeline.sh` 為入口，所有實驗參數寫在 `profiles/*.yaml`；新加實驗只要 `cp` 一份 YAML 改幾行，不必再改 Python 程式。

## 0. 環境與專案結構

### 0.1 必要條件

* 作業系統：Linux（範例路徑以 `/home/lawrencechh` 為主；WSL2 亦可）
* Python：3.10+（`uv run` 實際跑 3.12）
* 套件管理：`uv`
* PyYAML：已在 `uv` 環境內（讀取 profile YAML 用）
* Git：用來安裝 ComfyUI custom nodes
* 顯示卡：支援 Stable Diffusion / Illustrious 系列模型即可

### 0.2 專案目錄結構

```text
/home/lawrencechh/Lprojects/comfyui/
├── main.py
├── .venv/
├── models/
│   ├── checkpoints/        # 放 .safetensors 主模型
│   └── loras/              # 放 LoRA 檔
├── custom_nodes/
│   └── save-image-extended-comfyui/
├── output/                 # ComfyUI 預設輸出目錄
└── lawrence_tools/         # ← 你的工具集都在這
    ├── bash_pipeline.sh                       # 一條龍入口
    ├── gen_jsonl.py                           # 從 YAML/CLI 產 JSONL 任務列表
    ├── comfy_batch_generate_from_base.py      # 批次送 prompt + 搬檔案
    ├── profiles/                              # 每個實驗一份 YAML
    │   ├── kazuma.yaml
    │   └── endeavor.yaml
    └── workflows/                             # ComfyUI API 格式 workflow
        ├── base_workflow.json
        └── endeavor.json
```

> 重點：**workflow JSON 與 prompt / sweep 參數完全分離**。換 prompt 不用動 workflow；換 workflow（例如多掛一顆 LoRA）也不用改 Python — 程式會依 `class_type` 動態定位節點。

---

## 1. 安裝與設定 ComfyUI + SaveImageExtended

### 1.1 安裝 ComfyUI

```bash
cd /home/lawrencechh/Lprojects
git clone https://github.com/comfyanonymous/ComfyUI.git comfyui
cd comfyui

uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

啟動 ComfyUI：

```bash
cd /home/lawrencechh/Lprojects/comfyui
source .venv/bin/activate
uv run main.py --listen 127.0.0.1:8188
```

> `--listen 127.0.0.1:8188` 在本機開 HTTP API；Python 透過 `http://127.0.0.1:8188` 呼叫 `/prompt` 與 `/history`。

### 1.2 安裝 Save Image Extended custom node

不強制需要，GUI 測試時較好用。

```bash
cd /home/lawrencechh/Lprojects/comfyui/custom_nodes
git clone https://github.com/thedyze/save-image-extended-comfyui
```

重啟 ComfyUI 後，節點搜尋找得到 `Save Image Extended`。

---

## 2. 從 Civitai 取得模型與 Trigger Words

### 2.1 在 Civitai 確認 Base Model / Trigger Words

* **Base Model**：例如 `Illustrious` → 對應 `waiIllustriousSDXL_v80.safetensors`。
* **Trigger Words**：例如 `kazuma kiraboshi`，要放進正向 prompt。

### 2.2 下載並放到對的資料夾

```bash
mv waiIllustriousSDXL_v80.safetensors /home/lawrencechh/Lprojects/comfyui/models/checkpoints/
mv Kazuma_Kiraboshi_-_Illustrious.safetensors /home/lawrencechh/Lprojects/comfyui/models/loras/
```

回到 ComfyUI GUI 按 `R` 刷新模型列表。

---

## 3. 在 ComfyUI GUI 建立 workflow 並匯出 API JSON

### 3.1 GUI 內節點流（單 LoRA 範例）

```text
CheckpointLoaderSimple → LoraLoaderModelOnly → KSampler → VAEDecode → SaveImageExtended
EmptyLatentImage ──────────────────────────────→ KSampler
CLIPTextEncode (Positive) ─────────────────────→ KSampler.positive
CLIPTextEncode (Negative) ─────────────────────→ KSampler.negative
```

多 LoRA 只要把多個 `LoraLoaderModelOnly` 串成鏈：
`CheckpointLoaderSimple → LoraLoader#1 → LoraLoader#2 → KSampler` —— 程式會自動沿著 `model` 連線追溯整條 LoRA 鏈，**順序對應 `loras` list 的 index 0、1、…**。

關鍵：
- workflow 內各種參數值都會被 YAML / CLI 覆蓋，所以 GUI 內隨便填一組能跑的就好。
- `SaveImageExtended` 保持預設即可，**Python 程式不會去改它**。

### 3.2 匯出為 API JSON

GUI → Workflow → Export (API)，存到：
```
lawrence_tools/workflows/<檔名>.json
```

---

## 4. YAML profile 系統（主要操作介面）

每個實驗 = 一份 `profiles/<name>.yaml`。所有 sweep 參數、prompts、LoRA、workflow 路徑全寫在裡面。

### 4.1 YAML schema

```yaml
run_name: my_experiment                # 寫進 JSONL 與檔名前綴
workflow: workflows/base_workflow.json # 要載入的 ComfyUI API JSON
output: prompts_my_experiment.jsonl    # gen_jsonl 產出的 JSONL 路徑

image:
  width: 896
  height: 1344

sampling:
  steps: 28
  cfg_list: [6.0, 7.0]
  seed_range: [42, 45]                 # 包含兩端
  images_per_seed: 1
  sampler_list: [dpmpp_2m]             # 選填；不寫就沿用 workflow
  scheduler_list: [karras]             # 選填；不寫就沿用 workflow

# === LoRA 二擇一寫法 ===

# A) 單一固定 combo（多個 LoRA loader 都要列；順序＝Checkpoint → KSampler）
loras:
  - {name: Expressive_H-000001.safetensors, strength: 1.0}
  - {name: Endeavor_MHA-000009.safetensors, strength: 1.0}

# B) 多組 combo 做 sweep（每組可放多顆 LoRA）
lora_combos:
  - - {name: Kazuma.safetensors, strength: 0.5}
  - - {name: Kazuma.safetensors, strength: 0.7}
  - - {name: Kazuma.safetensors, strength: 0.9}

prompts:
  positive:
    - |
      prompt A（用 | 開頭可寫多行不用 escape）
    - |
      prompt B
  negative:
    - |
      neg A
    - |
      neg B
```

每個 positive × negative × cfg × sampler × scheduler × lora_combo × seed × image_index 都會展開成一個 job。

### 4.2 範例：`profiles/kazuma.yaml`（單 LoRA + strength sweep）

`lora_combos:` 列三組同名 LoRA 的不同 strength → 一次跑 0.5 / 0.7 / 0.9。

### 4.3 範例：`profiles/endeavor.yaml`（多 LoRA 固定組合）

`loras:` 列兩顆 LoRA（Expressive + Endeavor），程式會把第 0 顆掛到 workflow 中第 0 個 LoRA loader（最靠近 Checkpoint 那顆）。

### 4.4 動態節點偵測原理

`comfy_batch_generate_from_base.py` **不依賴硬編 node id**，而是：

1. 用 `class_type == "KSampler"` 找到 KSampler。
2. 沿 KSampler 的 `positive` / `negative` / `latent_image` 連線找到對應的 `CLIPTextEncode`、`EmptyLatentImage`。
3. 沿 KSampler 的 `model` 連線一路往回追溯 `LoraLoaderModelOnly` / `LoraLoader`，組出 LoRA chain。

所以你新匯出的 workflow 只要結構是「Checkpoint → (多顆) LoRA → KSampler」就能直接套用，**不用改 Python**。

### 4.5 CLI 覆寫（臨時改參數）

CLI > YAML > 預設。常用：

```bash
# 沿用 endeavor 設定，只改 seed-range 與 cfg
uv run gen_jsonl.py --config profiles/endeavor.yaml \
  --seed-range 100 105 --cfg-list 7.0 8.0
```

---

## 5. 三支腳本各自負責什麼

### 5.1 `gen_jsonl.py`

* **輸入**：YAML profile（`--config`）或一堆 CLI 參數。
* **輸出**：JSONL 任務列表（路徑來自 YAML 的 `output` 或 `--output`）。
* **額外功能**：`--print-field <key>` 印出 YAML 某個 top-level key 的值（給 `bash_pipeline.sh` 抓 `workflow` / `output` 路徑用）。

JSONL 每行欄位：

```json
{
  "id": "...",
  "run_name": "...",
  "prompt": "...",
  "negative": "...",
  "seed": 42,
  "steps": 28,
  "cfg": 6.0,
  "width": 832,
  "height": 1216,
  "loras": [{"name": "X.safetensors", "strength": 1.0}],
  "image_index": 1,
  "positive_index": 0,
  "negative_index": 0,
  "sampler_name": "dpmpp_2m",   // 選填
  "scheduler": "karras"          // 選填
}
```

向後相容：JSONL 若還是舊格式 `lora` / `lora_strength`（單一字串）也能讀。

### 5.2 `comfy_batch_generate_from_base.py`

* **輸入**：`--base-workflow workflows/X.json`、`--prompts X.jsonl`。
* **流程**：
  1. 讀 workflow，呼叫 `locate_workflow_slots()` 動態找 KSampler / CLIPTextEncode / EmptyLatentImage / LoRA chain。
  2. 對每個 job 把參數覆寫到 workflow，POST 到 `/prompt`。
  3. Poll `/history/{prompt_id}` 直到 `status.completed`。
  4. **從 history API 的 `outputs[*].images` 直接拿到實際的 filename + subfolder**（不再用檔案系統快照差分）。
  5. 把每張圖搬到 `--final-output-root / {run_name} /`，並重新命名為：
     ```
     {run_name}_loras{strengths}_cfg{cfg}_seed{seed}_img{N}_pp{positive_index}_np{negative_index}.png
     ```
     例如 `endeavor_run_loras1.0-1.0_cfg8.0_seed44_img1_pp4_np1.png`。
  6. 清理變空的 ComfyUI 原始子資料夾。

> 檔名裡的 `pp` / `np` 對應 YAML `prompts.positive` / `prompts.negative` 的 index — 重要：**這樣才能從檔名追回是哪段 prompt 產的**。

### 5.3 `bash_pipeline.sh`

一條龍入口，把上面兩支串起來。

```bash
PROFILE="${PROFILE:-kazuma}"
CONFIG="${CONFIG:-profiles/${PROFILE}.yaml}"
# 從 YAML 抓 workflow / 目標 JSONL 路徑
WORKFLOW=$(uv run --quiet gen_jsonl.py --config "$CONFIG" --print-field workflow)
PROMPTS=$(uv run --quiet gen_jsonl.py --config "$CONFIG" --print-field output)

uv run gen_jsonl.py --config "$CONFIG"
uv run comfy_batch_generate_from_base.py \
  --base-url "$BASE_URL" \
  --base-workflow "$WORKFLOW" \
  --prompts "$PROMPTS" \
  --comfy-output-root "$COMFY_OUTPUT_ROOT" \
  --final-output-root "$FINAL_OUTPUT_ROOT"
```

可覆寫的環境變數：`PROFILE`、`CONFIG`、`BASE_URL`、`COMFY_OUTPUT_ROOT`、`FINAL_OUTPUT_ROOT`。

---

## 6. 一條龍跑法

### 6.1 跑既有 profile

```bash
# 預設 (kazuma)
bash bash_pipeline.sh

# 切換 profile
PROFILE=endeavor bash bash_pipeline.sh

# 直接指定 config 檔（PROFILE 變數會被忽略）
CONFIG=profiles/my_new.yaml bash bash_pipeline.sh
```

### 6.2 新增一個實驗（人類動作）

1. **準備 workflow**：在 ComfyUI GUI 建好新 workflow，Export (API) 到 `workflows/<name>.json`。
2. **複製 profile**：`cp profiles/endeavor.yaml profiles/<name>.yaml`，修改 `workflow:` / `run_name:` / `output:` / `loras` / `prompts` / sweep 參數。
3. **啟動 ComfyUI**（如果還沒跑）。
4. **跑**：`PROFILE=<name> bash bash_pipeline.sh`。

---

## 7. 常見問題（Quick Debug）

### 7.1 「任務完成，但從 API 紀錄中沒有找到任何生成的圖片」

**症狀**：history 回報 `completed`，但 `image_files` 為空 / 路徑不存在。

**可能原因**：
- ComfyUI **快取**：seed + 所有節點輸入完全相同的 job 會被略過。雖然 history 還是會回上次的 filename，但若該檔早就被搬走 → `file_path.exists()` 為 False。
- workflow 沒有 SaveImage 節點：history outputs 不會有 `images`。

**解法**：
- 改 seed 或任何一個參數即可。
- 確認 workflow 含 `SaveImage` 或 `SaveImageExtended` 節點。

### 7.2 YAML 解析錯誤 `expected <block end>, but found '<block mapping start>'`

縮排不一致。YAML 同層級 key 縮排必須完全相同（建議全用 2 空格）。

### 7.3 ComfyUI 回傳 400 / 500

* `workflows/*.json` 是不是從 GUI 的 **Export API** 匯出（不是 workflow JSON）？兩種格式不同。
* 確認模型檔名拼字與 `models/checkpoints` / `models/loras` 內的實體一致。
* API body 應為 `{"prompt": { ... }}`（程式已自動包，手動測試需注意）。

### 7.4 LoRA 沒生效

* 開啟 ComfyUI log，確認 LoRA 被載入。
* 檢查 `profiles/X.yaml` 裡的 LoRA 名稱與 `models/loras/` 下完全一致（含副檔名）。
* `loras:` 順序對應 workflow 中 LoRA loader 由 Checkpoint 端開始的順序，若多顆 LoRA 反了會出現「掛的 LoRA 與預期相反」。

---

## 8. TL;DR：最短流程速查

```bash
# 1) 啟 ComfyUI（另開 terminal）
cd /home/lawrencechh/Lprojects/comfyui
source .venv/bin/activate
uv run main.py --listen 127.0.0.1:8188

# 2) 跑既有 profile
cd /home/lawrencechh/Lprojects/comfyui/lawrence_tools
PROFILE=endeavor bash bash_pipeline.sh

# (or) 臨時改 sweep 參數
uv run gen_jsonl.py --config profiles/endeavor.yaml \
  --seed-range 100 105 --cfg-list 7.0
uv run comfy_batch_generate_from_base.py \
  --base-workflow workflows/endeavor.json \
  --prompts prompts_endeavor.jsonl \
  --final-output-root /mnt/d/comfy_runs
```

---

## 9. 風格調整參數（給人看的直覺）

* **`lora_combos` strength 0.9 / 0.7 / 0.5**：
  * 0.9 → 最貼近 LoRA 訓練圖，風格嚴格。
  * 0.7 → 實用 sweet spot，角色辨識清楚但構圖留餘地。
  * 0.5 以下 → LoRA 變成微調，基底模型風格回來，角色更泛化。

* **`cfg_list` 8.0 / 7.0 / 6.0 / 5.0**：
  * 8.0 → 模型很聽 prompt，但容易僵硬、噪點多。
  * 7.0 / 6.0 → 畫質與創意的平衡區。
  * 5.0 → 模型較有想法，畫面自然但可能略偏 prompt。

* **Sweep 觀察策略**：
  * 第一輪先固定 seed，比較 `(LoRA strength × CFG)` 的格子。
  * 「嚴格 → 自由」大致是 `(0.9, 8.0) → (0.7, 7.0) → (0.5, 6.0) → (0.5, 5.0)`。

* **極端配方**：
  * 超嚴格：`lora strength 1.0 / 0.9 / 0.8` × `cfg 9.0 / 8.0 / 7.0`
  * 超自由：`lora strength 0.6 / 0.4 / 0.2` × `cfg 6.0 / 5.5 / 5.0`

> 建議習慣：每換一顆新 LoRA，先跑一輪 small grid，挑出最穩定的 1～2 組當該 LoRA 的預設參數，寫進它專屬的 `profiles/*.yaml`。
