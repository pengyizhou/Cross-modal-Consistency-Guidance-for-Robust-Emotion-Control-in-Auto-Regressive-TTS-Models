# CCG-CFG-TTS: Inference-Time Classifier-Free Guidance for Emotional TTS

This directory contains the **inference-time CFG (Classifier-Free Guidance)** component of [CCG-CFG-TTS](../README.md). It steers a pre-trained CosyVoice2 model toward a target emotion by contrasting conditional and unconditional generation paths at inference time, without additional fine-tuning.

The approach also includes an optional **learned CFG scale predictor** — a lightweight transformer that predicts token-level guidance scales from conditional/unconditional hidden states, replacing the need for a fixed scalar.

> For the **training-time DPO** component, see [cosyvoice-dpo/](../cosyvoice-dpo/).

---

## Directory Structure

```
CCG-CFG-TTS/
├── README.md
├── requirements.txt
├── cosyvoice/                          # CosyVoice2 with CFG modifications
│   ├── cli/cosyvoice.py                #   inference_instruct3 (single CFG), inference_instruct4 (opposite-instruction CFG)
│   ├── cli/model.py                    #   llm_job_cfg, CFG routing in tts()
│   ├── cli/frontend.py                 #   frontend_instruct3, frontend_instruct4
│   ├── llm/llm.py                      #   single/dual CFG in Qwen2LM.inference()
│   ├── models/cfg_predictor.py         #   learned CFG scale predictor (optional)
│   └── ...                             #   remaining CosyVoice2 modules
├── third_party/Matcha-TTS/             # Required on sys.path
├── configs/
│   └── cfg_predictor.yaml              # Config for CFG predictor training
├── scripts/                            # Evaluation utilities
│   ├── eval_gemini.py                  #   Model-as-Judge evaluation via Gemini API
│   ├── get_wer.py                      #   WhisperX-based WER
│   ├── compute-wer.py                  #   Kaldi-style WER computation
│   └── compute_dnsmos.py               #   DNSMOS quality score
├── tools/                              # External evaluation tools
│   ├── emotion2vec/                    #   Emotion recognition (bundled)
│   ├── setup_eval_tools.sh             #   Installs UTMOS, NISQA, etc.
│   └── UTMOS/, NISQA/                  #   Created by setup script
├── gen_audio_emotion_diff_reference/   # Inference pipelines + test data
│   ├── gen_dataset.py                  #   Audio generation script
│   ├── prepare_emotion_jsonl.py        #   JSONL preparation from Kaldi data
│   ├── gen_instruct_same_emo.sh        #   Baseline: same-emotion reference
│   ├── gen_instruct_neutral.sh         #   Baseline: neutral reference
│   ├── pipeline_cfg_gpt_condition_emotion.sh  # CFG sweep (scales 1.5–3.0)
│   └── data/
│       ├── test_jsonl/                 #   Test JSONL files + Kaldi test dir
│       └── utt2target_emo             #   Emotion mapping
└── gen_audio_emotion_diff_reference_train/  # Training data generation
    ├── gen_dataset.py
    ├── pipeline_cfg_gpt_condition_emotion_gt_ref.sh
    └── data/
        ├── train/                      #   Training JSONL files
        └── utt2target_emo
```

---

## Installation

### 1. Install dependencies

```bash
cd CCG-CFG-TTS
pip install -r requirements.txt
```

### 2. Download the CosyVoice2 base model

```bash
mkdir -p pretrained_models
# Download CosyVoice2-0.5B from ModelScope:
#   https://www.modelscope.cn/models/iic/CosyVoice2-0.5B
# Place the model files in pretrained_models/CosyVoice2-0.5B/
```

The model directory should contain: `cosyvoice.yaml`, `llm.pt`, `flow.pt`, `hift.pt`, `campplus.onnx`, `speech_tokenizer_v2.onnx`, `spk2info.pt`, and `CosyVoice-BlankEN/`.

### 3. Configure dataset paths

All data files (JSONL, `wav.scp`) ship with `{DATA_ROOT}` as a placeholder for your local dataset directory. Run the helper script to replace it with your actual path:

```bash
bash tools/update_data_paths.sh /path/to/your/datasets
```

Your datasets directory should contain:
- `emotion_dataset/ESD/Emotion_Speech_Dataset/` — [ESD](https://github.com/HLTSingapore/Emotional-Speech-Data) emotional speech corpus
- `libritts/` — [LibriTTS](https://www.openslr.org/60/) corpus
- `VCTK-Corpus/` — [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) corpus (used in training data only)

### 4. Set up evaluation tools

```bash
bash tools/setup_eval_tools.sh
```

This will:
- Clone and set up **UTMOS** (speech quality MOS predictor)
- Clone and set up **NISQA** (speech naturalness predictor)
- Install **funasr** for emotion2vec (emotion recognition)
- Install **whisperx** for WER evaluation
- Install **torchmetrics** for DNSMOS

After running the script, manually download the UTMOS checkpoints as instructed.

---

## Usage

All commands should be run from the `CCG-CFG-TTS/` directory.

### Baseline Generation (no CFG)

Generate speech with **same-emotion reference** audio (no guidance):

```bash
bash gen_audio_emotion_diff_reference/gen_instruct_same_emo.sh
```

Generate speech with **neutral reference** audio (no guidance):

```bash
bash gen_audio_emotion_diff_reference/gen_instruct_neutral.sh
```

### CFG-Guided Generation

Run CFG inference across multiple guidance scales (1.5, 2.0, 2.5, 3.0):

```bash
bash gen_audio_emotion_diff_reference/pipeline_cfg_gpt_condition_emotion.sh
```

### Training Data Generation with CFG

Generate training data with CFG guidance (scale=3.0, multiple random seeds). This data is used to construct DPO preference pairs for the [cosyvoice-dpo](../cosyvoice-dpo/) component:

```bash
bash gen_audio_emotion_diff_reference_train/pipeline_cfg_gpt_condition_emotion_gt_ref.sh
```

### Custom Single-Run Generation

```bash
python gen_audio_emotion_diff_reference/gen_dataset.py \
  --json_path gen_audio_emotion_diff_reference/data/test_jsonl/test.neutral.filter.jsonl \
  --model_path pretrained_models/CosyVoice2-0.5B \
  --save_path output_audio/ \
  --cfg_scale 2.5 \
  --drop_prompt 1 \
  --drop_target 0 \
  --target_emotion gen_audio_emotion_diff_reference/data/utt2target_emo
```

### Model-as-Judge Evaluation (Gemini)

Evaluate generated speech using Gemini as an LLM judge, scoring naturalness, emotional expressiveness, and overall quality on a 0–100 scale:

```bash
export GEMINI_API_KEY="your-api-key"

# Evaluate from a Kaldi-style directory
python scripts/eval_gemini.py --data-dir output_audio/kaldi_dir/

# Evaluate from a JSONL manifest
python scripts/eval_gemini.py --manifest output_audio/manifest.jsonl

# Evaluate from a tar archive (no disk extraction)
python scripts/eval_gemini.py --tar output_audio/samples.tar
```

Results are written to `auto_eval/gemini_judge.jsonl` (per-utterance) and `auto_eval/gemini_judge_summary.json` (aggregate). Use `--resume` to retry failed entries.

> **Cost estimate:** Evaluating each test set costs approximately **$27 USD** in Gemini API usage.

---

## Configuration

All pipeline scripts use environment variables for configurable paths. Set them before running, or edit the defaults at the top of each script:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `pretrained_models/CosyVoice2-0.5B` | Path to CosyVoice2 model |
| `EMOTION2VEC_DIR` | `tools/emotion2vec` | Path to emotion2vec scripts |
| `UTMOS_DIR` | `tools/UTMOS` | Path to UTMOS installation |
| `UTMOS_CKPT` | `tools/UTMOS/epoch=3-step=7459.ckpt` | UTMOS checkpoint path |
| `NISQA_DIR` | `tools/NISQA` | Path to NISQA installation |
| `DATA_DIR` | `gen_audio_emotion_diff_reference/data` | Test data directory |

---

## Data Format

Input JSONL files use the following schema:

```json
{
  "ID": "ESD-Emotion_Speech_Dataset-0011-Angry-0011_000372",
  "audio_path": "{DATA_ROOT}/emotion_dataset/ESD/Emotion_Speech_Dataset/0011/Angry/0011_000372.wav",
  "emotion": "angry",
  "content": "The transcript of the target utterance.",
  "ref_audio_path": "{DATA_ROOT}/libritts/train-clean-100/3879/174923/3879_174923_000034_000014.wav",
  "ref_emotion": "neutral",
  "ref_content": "The transcript of the reference utterance."
}
```

> **Note:** The provided data files use `{DATA_ROOT}` as a placeholder. Run `bash tools/update_data_paths.sh /your/data/root` to replace it with your local path (see [Installation step 3](#3-configure-dataset-paths)).

The `utt2target_emo` file is tab-separated: `utterance_id\tEmotion`.

---

## Evaluation Metrics

Each pipeline automatically runs the following evaluations:

| Metric | Tool | What it Measures |
|--------|------|-----------------|
| **Emotion Accuracy** | emotion2vec | Whether generated speech conveys the target emotion |
| **UTMOS** | UTMOS | Mean opinion score for speech quality (0–5) |
| **WER** | WhisperX | Word error rate vs. reference transcript |
| **NISQA** | NISQA | Speech naturalness (MOS) and quality dimensions |
| **DNSMOS** | torchmetrics | Deep noise suppression mean opinion score |
| **Gemini Judge** | eval_gemini.py | LLM-as-judge scoring: naturalness, emotional expressiveness, overall quality (0–100) |

---

## Model Checkpoints

All model checkpoints are referenced by download instructions and are **not** included in this repository.

| Model | How to Obtain | Size |
|-------|--------------|------|
| CosyVoice2-0.5B (base) | [ModelScope](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B) | ~2 GB |
| Fine-tuned CosyVoice2 | **Released after paper review** | TBD |
| emotion2vec_plus_large | Auto-downloaded via FunASR/ModelScope | ~300 MB |
| UTMOS | [HuggingFace](https://huggingface.co/spaces/sarulab-speech/UTMOS-demo) | ~2.2 GB |
| NISQA | [GitHub](https://github.com/gabrielmittag/NISQA) | ~2 MB |
| WhisperX large-v3-turbo | Auto-downloaded on first use | ~1.5 GB |

---

## Acknowledgements

This component builds upon:
- [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice) by Alibaba DAMO Academy
- [emotion2vec](https://github.com/ddlBoJack/emotion2vec)
- [UTMOS](https://github.com/sarulab-speech/UTMOS22)
- [NISQA](https://github.com/gabrielmittag/NISQA)
- [WhisperX](https://github.com/m-bain/whisperX)
