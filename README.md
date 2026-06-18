# CCG-CFG-TTS: Cross-modal Consistent Guided Classifier-Free Guidance for Emotional Text-to-Speech

[![arXiv](https://img.shields.io/badge/arXiv-2510.13293-b31b1b.svg)](https://arxiv.org/abs/2510.13293)
[![Demo Page](https://img.shields.io/badge/🎤%20Demo-Page-blue)](https://pengyizhou.github.io/Emotional_tts_demo/)
[![Interspeech 2026](https://img.shields.io/badge/Interspeech-2026-4CAF50?logo=academia)](https://arxiv.org/abs/2510.13293)
[![DOI](https://zenodo.org/badge/1173425478.svg)](https://doi.org/10.5281/zenodo.20741107)

This repository contains the official code for **CCG-CFG-TTS**, a framework for improving emotional expressiveness in text-to-speech (TTS) synthesis. The approach combines two complementary techniques applied to [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice):

1. **Classifier-Free Guidance (CFG) at inference time** — steers generation toward a target emotion by contrasting conditional and unconditional paths, requiring no additional fine-tuning.
2. **Direct Preference Optimization (DPO) at training time** — aligns the speech language model with preferences for emotional expressiveness using chosen/rejected sample pairs.

> **Model Weights:** We will release fine-tuned CosyVoice2 model weights after the paper review process. In the meantime, you can use the base [CosyVoice2-0.5B](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B) model from ModelScope.

---

## Repository Structure

```
emotion_tts_opensource/
├── README.md                         # This file
├── CCG-CFG-TTS/                      # Inference-time CFG guidance
│   ├── cosyvoice/                    #   CosyVoice2 with CFG modifications
│   ├── configs/                      #   CFG predictor training config
│   ├── gen_audio_emotion_diff_reference/       # Inference pipelines + test data
│   ├── gen_audio_emotion_diff_reference_train/ # Training data generation with CFG
│   ├── scripts/                      #   Evaluation utilities (WER, DNSMOS, Gemini judge)
│   ├── tools/                        #   emotion2vec, UTMOS, NISQA setup
│   └── third_party/Matcha-TTS/       #   Required dependency
│
└── cosyvoice-dpo/                    # Training-time DPO alignment
    ├── cosyvoice/                    #   CosyVoice2 with DPO modifications
    ├── conf/                         #   DPO training config
    ├── tools/                        #   DPO sample selection + data preparation
    ├── utils/                        #   Kaldi-style data utilities
    ├── run_dpo.sh                    #   Full DPO training pipeline
    └── third_party/Matcha-TTS/       #   Required dependency
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA 12.1+ (for GPU inference and training)
- ~10 GB disk space for pretrained models and evaluation tools

### 1. Download the base model

Download [CosyVoice2-0.5B](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B) and place it under each component's `pretrained_models/` directory:

```bash
# For CFG inference
mkdir -p CCG-CFG-TTS/pretrained_models/CosyVoice2-0.5B/
# For DPO training
mkdir -p cosyvoice-dpo/pretrained_models/CosyVoice2-0.5B/
```

The model directory should contain: `cosyvoice.yaml`, `llm.pt`, `flow.pt`, `hift.pt`, `campplus.onnx`, `speech_tokenizer_v2.onnx`, `spk2info.pt`, and `CosyVoice-BlankEN/`.

### 2. CFG-guided emotional speech generation (inference)

```bash
cd CCG-CFG-TTS
pip install -r requirements.txt

# Configure dataset paths
bash tools/update_data_paths.sh /path/to/your/datasets

# Generate speech with CFG guidance at multiple scales
bash gen_audio_emotion_diff_reference/pipeline_cfg_gpt_condition_emotion.sh
```

See [CCG-CFG-TTS/README.md](CCG-CFG-TTS/README.md) for the full guide on baselines, CFG sweeps, the learned CFG predictor, and evaluation.

### 3. DPO training for emotional alignment

```bash
cd cosyvoice-dpo
pip install -r requirements.txt
cd third_party/Matcha-TTS && pip install -e . && cd ../..

# Configure paths in run_dpo.sh, then run the full pipeline
bash run_dpo.sh  # with stage=0 stop_stage=10
```

See [cosyvoice-dpo/README.md](cosyvoice-dpo/README.md) for the full guide on DPO sample selection, feature extraction, and training.

---

## Datasets

The following publicly available datasets are used:

| Dataset | Usage | Link |
|---------|-------|------|
| **ESD** (Emotional Speech Dataset) | Emotional speech corpus for evaluation and training | [GitHub](https://github.com/HLTSingapore/Emotional-Speech-Data) |
| **LibriTTS** | Neutral reference speech | [OpenSLR](https://www.openslr.org/60/) |
| **VCTK** | Additional training data | [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3443) |

---

## Evaluation Metrics

Both components share a common evaluation suite:

| Metric | Tool | Measures |
|--------|------|----------|
| **Emotion Accuracy** | emotion2vec | Whether generated speech conveys the target emotion |
| **UTMOS** | UTMOS | Mean opinion score for speech quality (0–5) |
| **WER** | WhisperX | Word error rate vs. reference transcript |
| **NISQA** | NISQA | Speech naturalness and quality dimensions |
| **DNSMOS** | torchmetrics | Deep noise suppression mean opinion score |
| **Gemini Judge** | eval_gemini.py | LLM-as-judge: naturalness, emotional expressiveness, overall quality (0–100) |

Evaluation tools are set up via:

```bash
cd CCG-CFG-TTS
bash tools/setup_eval_tools.sh
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{peng2026ccgcfgtts,
  title={Cross-modal Consistency Guidance for Robust Emotion Control in Auto-Regressive TTS Models},
  author={Peng, Yizhou and Ma, Yukun and Zhang, Chong and Chao, Yi-Wen and Ni, Chongjia and Ma, Bin and Chng, Eng Siong},
  booktitle={Proceedings of Interspeech 2026},
  year={2026},
  eprint={2510.13293},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  doi={10.48550/arXiv.2510.13293}
}
```

---

## Acknowledgements

This project builds upon:
- [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice) by Alibaba DAMO Academy
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)
- [emotion2vec](https://github.com/ddlBoJack/emotion2vec)
- [UTMOS](https://github.com/sarulab-speech/UTMOS22)
- [NISQA](https://github.com/gabrielmittag/NISQA)
- [WhisperX](https://github.com/m-bain/whisperX)

## License

This codebase is released under the NTUitive license. Third-party components retain their original licenses.
