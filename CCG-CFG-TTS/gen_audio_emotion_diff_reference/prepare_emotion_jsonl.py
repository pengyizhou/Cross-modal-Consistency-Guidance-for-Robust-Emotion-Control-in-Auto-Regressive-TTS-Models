#!/usr/bin/env python3
"""
Generate JSONL from Kaldi-style split_emotion data (wav.scp, text, utt2text_emo/utt2target_emo).
Each row has content from text, audio_path from wav.scp, emotion chosen from ALLOWED_EMOTIONS
excluding the emotion in utt2text_emo, and ref_audio_path/ref_emotion/ref_content from the
best reference log for the selected emotion.
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

ALLOWED_EMOTIONS = [
    "Neutral",
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Disgusted",
    "Surprised",
]

# Emotion subdirs expected under data_root
EMOTION_SUBDIRS = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]


def parse_reference_log(reference_log: Path) -> Dict[str, Dict[str, str]]:
    """
    Parse best_reference_audio.log into {emotion_lower: {"path": ..., "text": ...}}.
    """
    references = {}
    current_emotion = None
    current_data = {}

    with open(reference_log, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            # Match "emotion:" at start of line (e.g., "angry:")
            emo_match = re.match(r"^(\w+)\s*:\s*$", line)
            if emo_match:
                # Save previous block if any
                if current_emotion and "path" in current_data and "text" in current_data:
                    references[current_emotion.lower()] = current_data

                current_emotion = emo_match.group(1)
                current_data = {}
                continue

            # Match "  text: ..." or "  path: ..."
            text_match = re.match(r"^\s*text\s*:\s*(.+)$", line)
            path_match = re.match(r"^\s*path\s*:\s*(.+)$", line)
            if text_match:
                current_data["text"] = text_match.group(1).strip()
            elif path_match:
                current_data["path"] = path_match.group(1).strip()

        # Don't forget last block
        if current_emotion and "path" in current_data and "text" in current_data:
            references[current_emotion.lower()] = current_data

    return references


def load_kaldi_file(path: Path, key_value_sep: Optional[str] = None) -> Dict[str, str]:
    """
    Load a Kaldi-style file (wav.scp, text, utt2*) as {utt_id: value}.
    Uses first whitespace as separator; value is the rest of the line.
    """
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            result[parts[0]] = parts[1]
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate JSONL from Kaldi split_emotion data with best reference samples."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Kaldi data root (split_emotion) with emotion subdirs (Angry, Neutral, etc.).",
    )
    parser.add_argument(
        "--reference_log",
        type=Path,
        required=True,
        help="Path to best_reference_audio.log",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--emotion_file",
        type=str,
        default="utt2text_emo",
        choices=["utt2text_emo", "utt2target_emo"],
        help="Which file to use for emotion exclusion (default: utt2text_emo)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Restrict to specific emotion subdirs (e.g., Angry Neutral). If not set, process all.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Parse reference log
    references = parse_reference_log(args.reference_log)
    if not references:
        raise ValueError(f"No valid reference blocks found in {args.reference_log}")

    # Ensure all ALLOWED_EMOTIONS have references (lowercase keys)
    for emo in ALLOWED_EMOTIONS:
        key = emo.lower()
        if key not in references:
            raise ValueError(
                f"Reference log missing emotion '{key}'. "
                f"Available: {list(references.keys())}"
            )

    # Determine which subdirs to process
    subdirs = args.splits if args.splits else EMOTION_SUBDIRS

    rows = []
    skipped_empty_allowed = 0
    skipped_missing_ref = 0

    for emo_subdir in subdirs:
        subdir_path = args.data_root / emo_subdir
        if not subdir_path.is_dir():
            continue

        wav_scp_path = subdir_path / "wav.scp"
        text_path = subdir_path / "text"
        emo_path = subdir_path / args.emotion_file

        if not wav_scp_path.exists() or not text_path.exists() or not emo_path.exists():
            continue

        wav_scp = load_kaldi_file(wav_scp_path)
        text = load_kaldi_file(text_path)
        utt2emo = load_kaldi_file(emo_path)

        # Iterate over utterances present in all three
        utt_ids = set(wav_scp.keys()) & set(text.keys()) & set(utt2emo.keys())

        for utt_id in utt_ids:
            content = text[utt_id]
            audio_path = wav_scp[utt_id]
            exclude_emo_raw = utt2emo[utt_id]
            exclude_emo = exclude_emo_raw.strip().lower()

            allowed = [
                e for e in ALLOWED_EMOTIONS
                if e.lower() != exclude_emo
            ]

            if not allowed:
                skipped_empty_allowed += 1
                continue

            selected = random.choice(allowed)
            ref_emotion = selected.lower()
            ref = references.get(ref_emotion)

            if ref is None:
                skipped_missing_ref += 1
                continue

            rows.append({
                "ID": utt_id,
                "audio_path": audio_path,
                "emotion": ref_emotion,
                "content": content,
                "ref_audio_path": ref["path"],
                "ref_emotion": ref_emotion,
                "ref_content": ref["text"],
            })

    if skipped_empty_allowed:
        print(f"Warning: Skipped {skipped_empty_allowed} utterances with empty allowed emotion set")
    if skipped_missing_ref:
        print(f"Warning: Skipped {skipped_missing_ref} utterances due to missing reference")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
