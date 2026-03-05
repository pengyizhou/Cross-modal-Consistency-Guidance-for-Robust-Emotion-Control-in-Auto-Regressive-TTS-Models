#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Setting up evaluation tools in: $TOOLS_DIR ==="

# -----------------------------------------------------------------------
# 1. UTMOS  (speech quality MOS predictor)
# -----------------------------------------------------------------------
if [ ! -d "$TOOLS_DIR/UTMOS" ]; then
    echo ""
    echo "--- Cloning UTMOS ---"
    git clone https://github.com/sarulab-speech/UTMOS22.git "$TOOLS_DIR/UTMOS"
    cd "$TOOLS_DIR/UTMOS"

    echo "--- Downloading UTMOS checkpoints ---"
    # Main checkpoint
    if [ ! -f "epoch=3-step=7459.ckpt" ]; then
        echo "Please download epoch=3-step=7459.ckpt from:"
        echo "  https://huggingface.co/spaces/sarulab-speech/UTMOS-demo/tree/main"
        echo "and place it in: $TOOLS_DIR/UTMOS/"
    fi
    # wav2vec_small.pt (used by UTMOS internally)
    if [ ! -f "wav2vec_small.pt" ]; then
        echo "Please download wav2vec_small.pt from:"
        echo "  https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt"
        echo "and place it in: $TOOLS_DIR/UTMOS/"
    fi

    pip install pytorch-lightning fairseq 2>/dev/null || echo "  (install pytorch-lightning and fairseq manually if needed)"
    cd "$TOOLS_DIR"
else
    echo "UTMOS already exists, skipping."
fi

# -----------------------------------------------------------------------
# 2. NISQA  (speech naturalness / quality predictor)
# -----------------------------------------------------------------------
if [ ! -d "$TOOLS_DIR/NISQA" ]; then
    echo ""
    echo "--- Cloning NISQA ---"
    git clone https://github.com/gabrielmittag/NISQA.git "$TOOLS_DIR/NISQA"
    cd "$TOOLS_DIR/NISQA"

    pip install -e . 2>/dev/null || echo "  (install NISQA manually: cd tools/NISQA && pip install -e .)"
    cd "$TOOLS_DIR"
else
    echo "NISQA already exists, skipping."
fi

# -----------------------------------------------------------------------
# 3. emotion2vec  (bundled in tools/emotion2vec/, just install funasr)
# -----------------------------------------------------------------------
echo ""
echo "--- Installing funasr for emotion2vec ---"
pip install -U funasr 2>/dev/null || echo "  (install funasr manually: pip install -U funasr)"

# -----------------------------------------------------------------------
# 4. WhisperX  (ASR for WER evaluation)
# -----------------------------------------------------------------------
echo ""
echo "--- Installing whisperx ---"
pip install whisperx 2>/dev/null || echo "  (install whisperx manually: pip install whisperx)"

# -----------------------------------------------------------------------
# 5. DNSMOS dependencies
# -----------------------------------------------------------------------
echo ""
echo "--- Installing DNSMOS dependencies ---"
pip install "torchmetrics[audio]" librosa 2>/dev/null || echo "  (install manually: pip install 'torchmetrics[audio]' librosa)"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Remaining manual steps:"
echo "  1. Download UTMOS checkpoints (see messages above)"
echo "  2. Download CosyVoice2-0.5B model from ModelScope into pretrained_models/"
echo "  3. See README.md for full instructions"
