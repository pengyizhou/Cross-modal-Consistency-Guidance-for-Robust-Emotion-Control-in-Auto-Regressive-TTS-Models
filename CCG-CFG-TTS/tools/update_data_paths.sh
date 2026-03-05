#!/bin/bash
# Replace {DATA_ROOT} placeholder in all data files with your local dataset path.
#
# Usage:
#   bash tools/update_data_paths.sh /path/to/your/datasets
#
# Your datasets directory should contain:
#   emotion_dataset/ESD/Emotion_Speech_Dataset/  (ESD emotional speech corpus)
#   libritts/                                     (LibriTTS corpus)
#   VCTK-Corpus/                                  (VCTK corpus, optional, used in training data)

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <data_root>"
    echo "  <data_root>  Absolute path to the directory containing emotion_dataset/, libritts/, VCTK-Corpus/"
    echo ""
    echo "Example: $0 /data/speech/datasets"
    exit 1
fi

DATA_ROOT="${1%/}"

if [[ "$DATA_ROOT" != /* ]]; then
    echo "Error: <data_root> must be an absolute path (starting with /)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Replacing {DATA_ROOT} -> $DATA_ROOT in all data files..."

find "$SCRIPT_DIR/gen_audio_emotion_diff_reference/data" \
     "$SCRIPT_DIR/gen_audio_emotion_diff_reference_train/data" \
     -type f \( -name '*.jsonl' -o -name 'wav.scp' \) \
     -exec sed -i "s|{DATA_ROOT}|${DATA_ROOT}|g" {} +

echo "Done. Replaced paths in:"
find "$SCRIPT_DIR/gen_audio_emotion_diff_reference/data" \
     "$SCRIPT_DIR/gen_audio_emotion_diff_reference_train/data" \
     -type f \( -name '*.jsonl' -o -name 'wav.scp' \) -print
