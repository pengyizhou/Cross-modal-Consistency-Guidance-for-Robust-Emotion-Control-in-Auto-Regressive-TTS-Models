#!/usr/bin/env python3
"""
Model-as-Judge TTS evaluation using Gemini via Vector Engine REST API.

Evaluates synthesised speech on three dimensions per utterance:
  1. Naturalness             – prosody, rhythm, intonation           (0–100)
  2. Emotional Expressiveness – match to a provided target emotion   (0–100)
  3. Overall Quality          – intelligibility / clarity            (0–100)
  + Final Score              – average of the three dimensions       (0–100)

Supported input formats
-----------------------
  --data-dir  DIR    Kaldi-style directory that contains:
                       wav.scp        (utt_id  audio_path)
                       utt2emotion    (utt_id  emotion_label)   [optional]
                       text           (utt_id  transcript)      [optional]

  --tar       FILE   Tar archive containing wav.scp (uttid ./uttid.wav),
                       utt2emotion, text, and *.wav files. Loads directly into
                       memory (no disk extraction).

  --manifest  FILE   JSONL file, one item per line, with keys:
                       audio_path     (required)
                       target_emotion (optional)
                       text           (optional)
                       utt_id         (optional; derived from audio_path if absent)

Outputs
-------
  --output   FILE   Per-utterance JSONL with scores + rationale
  --summary  FILE   Aggregate summary JSON (mean ± std per dimension)

Install
-------
  pip install requests
"""

import argparse
import base64
import json
import math
import os
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import requests

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PROMPT = """\
You are an expert speech quality evaluator with extensive experience in \
text-to-speech (TTS) assessment. Listen carefully to the synthesised speech \
audio sample below and evaluate it on exactly three dimensions.

For each dimension assign an integer score from 0 to 100 using the criteria \
below. Use the full range — reserve scores near 100 for genuinely excellent \
samples and scores near 0 for severely defective ones.

──────────────────────────────────────────────────────────────────────────────
1. NATURALNESS
   How natural does the speech sound overall? Consider prosody, rhythm, \
intonation, speaking rate, and the absence of artefacts.
   90–100 – Completely natural; indistinguishable from fluent human speech.
   70–89  – Mostly natural; only minor unnatural inflections or artefacts.
   50–69  – Somewhat natural; noticeable but tolerable prosodic issues.
   25–49  – Unnatural; frequent robotic, monotone, or broken prosody.
   0–24   – Very unnatural; severe artefacts or clearly machine-generated.

──────────────────────────────────────────────────────────────────────────────
2. EMOTIONAL EXPRESSIVENESS
   How convincingly does the speech convey the specified TARGET EMOTION?
   90–100 – The target emotion is clearly and consistently expressed throughout.
   70–89  – The target emotion is evident and mostly consistent.
   50–69  – The target emotion is perceptible but weak or intermittent.
   25–49  – The target emotion is barely perceptible.
   0–24   – The target emotion is absent or an incorrect emotion is conveyed.

   If no target emotion is specified, score based on whether the speech \
expresses any coherent emotion appropriate for the spoken content.

──────────────────────────────────────────────────────────────────────────────
3. OVERALL QUALITY (INTELLIGIBILITY)
   How clearly and intelligibly are the words articulated?
   90–100 – Perfectly clear; every word is fully intelligible.
   70–89  – Mostly clear; rare moments of unclear articulation.
   50–69  – Generally understandable; occasional unclear or mispronounced words.
   25–49  – Difficult to understand; frequent unclear or garbled passages.
   0–24   – Very poor; largely unintelligible.

──────────────────────────────────────────────────────────────────────────────
Compute the FINAL SCORE as the simple arithmetic average of the three \
dimensions (rounded to one decimal place).

RESPONSE FORMAT (strictly JSON, no other text):
{
  "naturalness": <integer 0-100>,
  "naturalness_rationale": "<one concise sentence>",
  "emotional_expressiveness": <integer 0-100>,
  "emotional_expressiveness_rationale": "<one concise sentence>",
  "overall_quality": <integer 0-100>,
  "overall_quality_rationale": "<one concise sentence>",
  "final_score": <number, one decimal place>
}
"""

# ---------------------------------------------------------------------------
# Data loading  (identical to eval_tts.py)
# ---------------------------------------------------------------------------

def _read_kaldi_map(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    return result


def load_kaldi_dir(data_dir: str) -> List[Dict]:
    d = Path(data_dir)
    wav_scp = d / "wav.scp"
    assert wav_scp.exists(), f"wav.scp not found in {data_dir}"

    wav_map = _read_kaldi_map(wav_scp)
    wav_base = wav_scp.parent
    emo_map = _read_kaldi_map(d / "utt2emotion") if (d / "utt2emotion").exists() else {}
    text_map = _read_kaldi_map(d / "text") if (d / "text").exists() else {}

    entries = []
    for utt_id, audio_path in wav_map.items():
        resolved = Path(audio_path)
        if not resolved.is_absolute():
            resolved = (wav_base / resolved).resolve()
        entries.append({
            "utt_id": utt_id,
            "audio_path": str(resolved),
            "target_emotion": emo_map.get(utt_id),
            "text": text_map.get(utt_id),
        })
    return entries


def load_manifest(manifest_path: str) -> List[Dict]:
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            audio_path = item.get("audio_path", "")
            utt_id = item.get("utt_id") or Path(audio_path).stem
            entries.append({
                "utt_id": utt_id,
                "audio_path": audio_path,
                "target_emotion": item.get("target_emotion"),
                "text": item.get("text"),
            })
    return entries


def _read_tar_text_member(tf: tarfile.TarFile, name: str) -> str:
    """Read a text file member from tar. Returns empty string if not found."""
    try:
        m = tf.getmember(name)
    except KeyError:
        return ""
    return tf.extractfile(m).read().decode("utf-8")  # type: ignore


def load_tar(tar_path: str) -> List[Dict]:
    """Load tar archive directly into memory (no disk extraction).

    Tar must contain wav.scp (uttid ./uttid.wav), utt2emotion, text, and *.wav.
    Reads all files from the tar and builds entries with audio_bytes pre-loaded.
    """
    tp = Path(tar_path)
    assert tp.exists(), f"Tar file not found: {tar_path}"

    def parse_kaldi_text(content: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
        return result

    entries: List[Dict] = []
    with tarfile.open(tp, "r:*") as tf:
        member_names = {m.name for m in tf.getmembers()}
        wav_scp_raw = _read_tar_text_member(tf, "wav.scp")
        if not wav_scp_raw:
            raise ValueError("wav.scp not found in tar")
        emo_raw = _read_tar_text_member(tf, "utt2emotion")
        text_raw = _read_tar_text_member(tf, "text")

        wav_map = parse_kaldi_text(wav_scp_raw)
        emo_map = parse_kaldi_text(emo_raw)
        text_map = parse_kaldi_text(text_raw)

        for utt_id, audio_path in wav_map.items():
            # Path like ./uttid.wav -> wav file in tar is uttid.wav (basename)
            wav_name = Path(audio_path).name
            if wav_name not in member_names:
                # Fallback: uttid.wav if path format differs
                wav_name = f"{utt_id}.wav"
            if wav_name not in member_names:
                raise ValueError(f"WAV file {wav_name} (for {utt_id}) not found in tar")
            m = tf.getmember(wav_name)
            audio_bytes = tf.extractfile(m).read()  # type: ignore
            mime = _MIME_MAP.get(Path(wav_name).suffix.lower(), "audio/wav")

            entries.append({
                "utt_id": utt_id,
                "audio_bytes": audio_bytes,
                "mime_type": mime,
                "target_emotion": emo_map.get(utt_id),
                "text": text_map.get(utt_id),
            })

    return entries

# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

_MIME_MAP = {
    ".wav":  "audio/wav",
    ".mp3":  "audio/mp3",
    ".flac": "audio/flac",
    ".ogg":  "audio/ogg",
    ".m4a":  "audio/mp4",
}


def load_audio(audio_path: str) -> Tuple[bytes, str]:
    """Return (raw_bytes, mime_type) for the given audio file."""
    path = Path(audio_path)
    assert path.exists(), f"Audio file not found: {audio_path}"
    mime = _MIME_MAP.get(path.suffix.lower(), "audio/wav")
    with open(path, "rb") as f:
        return f.read(), mime

# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


def _extract_eval_json_from_response(data: Dict) -> str:
    """Extract evaluation JSON from Gemini response, handling thought/reasoning parts."""
    parts = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    )
    if not parts:
        raise ValueError("Empty response; no parts in candidate")

    # Prefer part with thoughtSignature (final answer); skip parts with thought=True
    for part in parts:
        if part.get("thought") is True:
            continue
        text = part.get("text", "").strip()
        if not text:
            continue
        # Prefer thoughtSignature part; otherwise accept first valid-looking JSON
        if "thoughtSignature" in part or text.strip().startswith("{"):
            return text
    # Fallback: last non-thought part
    for part in reversed(parts):
        if part.get("thought") is True:
            continue
        text = part.get("text", "").strip()
        if text:
            return text
    raise ValueError("No evaluable text in response parts")


def build_contents(entry: Dict, audio_bytes: bytes, mime_type: str) -> Dict:
    """Build the contents dict for one generate_content REST call."""
    lines = [PROMPT, "\n── This utterance ──"]
    if entry.get("target_emotion"):
        lines.append(f"Target emotion: {entry['target_emotion']}")
    if entry.get("text"):
        lines.append(f"Reference transcript: {entry['text']}")
    lines.append("\nListen to the audio and return your JSON evaluation.")

    return {
        "role": "user",
        "parts": [
            {"text": "\n".join(lines)},
            {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(audio_bytes).decode("ascii")}},
        ],
    }


def call_judge(
    entry: Dict,
    api_key: str,
    base_url: str,
    model: str = "gemini-3.1-pro-preview",
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> Dict:
    result: Dict = {"utt_id": entry["utt_id"]}
    if entry.get("target_emotion"):
        result["target_emotion"] = entry["target_emotion"]
    if entry.get("text"):
        result["text"] = entry["text"]

    if "audio_bytes" in entry and "mime_type" in entry:
        audio_bytes, mime_type = entry["audio_bytes"], entry["mime_type"]
    else:
        audio_bytes, mime_type = load_audio(entry["audio_path"])
    contents = build_contents(entry, audio_bytes, mime_type)

    url = f"{base_url.rstrip('/')}/v1beta/models/{model}:generateContent"
    params = {} if not api_key else {"key": api_key}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-goog-api-key"] = api_key

    payload = {
        "contents": [contents],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }

    _REQUIRED_KEYS = {"naturalness", "emotional_expressiveness", "overall_quality"}
    raw = ""

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, params=params, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429 or (resp.status_code >= 500 and attempt < max_retries):
                time.sleep(3)
                continue
            resp.raise_for_status()

            data = resp.json()
            raw = _extract_eval_json_from_response(data)

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            scores = json.loads(raw)

            if not _REQUIRED_KEYS.issubset(
                {k for k in scores if isinstance(scores[k], (int, float))}
            ):
                raise ValueError(f"Missing required score keys; got: {list(scores.keys())}")

            result.update(scores)
            dim_keys = ["naturalness", "emotional_expressiveness", "overall_quality"]
            result["final_score"] = round(sum(scores[k] for k in dim_keys) / 3, 1)
            return result

        except requests.RequestException as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "quota" in err_str.lower() or "exhausted" in err_str.lower()
            if is_rate_limit and attempt < max_retries:
                time.sleep(3)
                continue
            result["error"] = f"Request error: {e}"
            return result

        except (json.JSONDecodeError, ValueError) as e:
            if attempt < max_retries:
                time.sleep(3)
                continue
            result["error"] = f"Bad response after {max_retries} attempts: {e} | raw={raw!r}"
            return result

        except Exception as e:  # noqa: BLE001
            result["error"] = str(e)
            return result

    return result

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

SCORE_KEYS = ["naturalness", "emotional_expressiveness", "overall_quality", "final_score"]


def summarize(items: List[Dict]) -> Dict:
    from collections import defaultdict
    acc = defaultdict(list)
    error_count = 0
    for item in items:
        if item.get("error"):
            error_count += 1
            continue
        for k in SCORE_KEYS:
            if k in item and isinstance(item[k], (int, float)):
                acc[k].append(float(item[k]))

    summary: Dict = {"total": len(items), "errors": error_count}
    for k, vals in acc.items():
        n = len(vals)
        mean = sum(vals) / n if n else float("nan")
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n > 1 else 0.0
        summary[k] = {"mean": round(mean, 4), "std": round(std, 4), "n": n}
    return summary

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Model-as-Judge TTS evaluation via Vector Engine REST API"
    )
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--data-dir", type=str, help="Kaldi-style data directory")
    inp.add_argument("--tar", type=str, help="Tar archive (wav.scp, utt2emotion, text, *.wav)")
    inp.add_argument("--manifest", type=str, help="JSONL manifest file")

    p.add_argument("--output",  type=str, default="auto_eval/gemini_judge.jsonl")
    p.add_argument("--summary", type=str, default="auto_eval/gemini_judge_summary.json")

    p.add_argument("--model",   type=str, default="gemini-3.1-pro-preview")
    p.add_argument("--api-key", type=str, default=None,
                   help="Gemini API key (default: $GEMINI_API_KEY env var)")
    p.add_argument("--base-url", type=str, default=None,
                   help="API base URL (default: https://generativelanguage.googleapis.com)")

    p.add_argument("--workers",     type=int, default=3)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--resume",      action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.data_dir:
        entries = load_kaldi_dir(args.data_dir)
        print(f"Loaded {len(entries)} utterances from {args.data_dir}")
    elif args.tar:
        entries = load_tar(args.tar)
        print(f"Loaded {len(entries)} utterances from {args.tar} (in-memory)")
    else:
        entries = load_manifest(args.manifest)
        print(f"Loaded {len(entries)} utterances from {args.manifest}")

    output_path = Path(args.output)
    done_ids: set = set()
    done_records: List[Dict] = []  # when resume: loaded from output
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        done_ids.add(rec["utt_id"])
                        done_records.append(rec)
                    except (json.JSONDecodeError, KeyError):
                        pass
        failed_ids = {r["utt_id"] for r in done_records if r.get("error")}
        if failed_ids:
            # Resume mode: only run failed entries
            entries = [e for e in entries if e["utt_id"] in failed_ids]
            print(f"Resuming: re-running {len(entries)} failed entries (from {len(done_ids)} total)")
        else:
            # No failures in output; nothing to run
            print(f"Resuming: no failed entries in output ({len(done_ids)} total succeeded). Nothing to run.")
            return

    if args.max_samples is not None:
        entries = entries[: args.max_samples]
        print(f"Limiting to {len(entries)} utterances (--max-samples)")

    if not entries:
        print("No utterances to evaluate. Exiting.")
        return

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    base_url = (args.base_url or "https://generativelanguage.googleapis.com").rstrip("/")
    if args.base_url:
        print(f"Using custom base URL: {base_url}")

    judge_fn = partial(
        call_judge,
        api_key=api_key,
        base_url=base_url,
        model=args.model,
        max_retries=args.max_retries,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path = output_path.parent / (output_path.stem + "_error" + output_path.suffix)
    results: List[Dict] = []
    error_results: List[Dict] = []
    is_resume = args.resume and bool(done_records)

    print(f"Evaluating {len(entries)} utterances with model={args.model}, "
          f"workers={args.workers} …")

    out_f = open(output_path, "w", encoding="utf-8") if not is_resume else None
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(judge_fn, e): e for e in entries}
            completed = 0
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {"utt_id": entry["utt_id"], "error": str(exc)}
                results.append(result)
                if out_f is not None:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
                completed += 1
                status = "OK" if not result.get("error") else f"ERR: {result['error'][:60]}"
                print(f"  [{completed}/{len(entries)}] {result['utt_id'][:60]} → {status}")
    finally:
        if out_f is not None:
            out_f.close()

    if is_resume:
        # Merge: replace failed records in done_records with new results, then overwrite output
        new_by_id = {r["utt_id"]: r for r in results}
        merged_records = [new_by_id.get(r["utt_id"], r) for r in done_records]
        with output_path.open("w", encoding="utf-8") as out_f:
            for r in merged_records:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        results = merged_records

    # Retry failed entries after all others finished (same max_retries per entry)
    error_results = [r for r in results if r.get("error")]
    if error_results:
        failed_ids = {r["utt_id"] for r in error_results}
        failed_entries = [e for e in entries if e["utt_id"] in failed_ids]
        print(f"\nRetrying {len(failed_entries)} failed entries …")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(judge_fn, e): e for e in failed_entries}
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {"utt_id": entry["utt_id"], "error": str(exc)}
                if result.get("error"):
                    error_results = [r for r in error_results if r["utt_id"] != result["utt_id"]]
                    error_results.append(result)
                else:
                    # Succeeded on retry: replace in results, remove from error list
                    results = [result if r["utt_id"] == result["utt_id"] else r for r in results]
                    error_results = [r for r in error_results if r["utt_id"] != result["utt_id"]]
                    print(f"  Retry OK: {result['utt_id'][:60]}")

        # Rewrite output with final results (replacements from retry)
        with output_path.open("w", encoding="utf-8") as out_f:
            for r in results:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # Write error file with only entries that still fail after retries
        with error_path.open("w", encoding="utf-8") as err_f:
            for r in error_results:
                err_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {len(error_results)} entries still failed → {error_path}")

    summary = summarize(results)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results written to {output_path}")
    if error_results:
        print(f"✓ Error entries written to {error_path}")
    print(f"✓ Summary written to {summary_path}")
    print("\n── Summary (0–100 scale) ──")
    dim_keys = ["naturalness", "emotional_expressiveness", "overall_quality"]
    for k in dim_keys:
        if k in summary:
            s = summary[k]
            print(f"  {k:30s}  mean={s['mean']:.1f}  std={s['std']:.1f}  n={s['n']}")
    if "final_score" in summary:
        s = summary["final_score"]
        print(f"  {'─' * 44}")
        print(f"  {'final_score (avg)':30s}  mean={s['mean']:.1f}  std={s['std']:.1f}  n={s['n']}")
    if summary.get("errors", 0):
        print(f"  Errors: {summary['errors']}")


if __name__ == "__main__":
    main()
