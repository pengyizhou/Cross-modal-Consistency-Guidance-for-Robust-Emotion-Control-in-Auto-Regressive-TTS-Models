#!/usr/bin/env python3
"""
Compute DNSMOS (Deep Noise Suppression Mean Opinion Score) for WAV files
using torchmetrics. Outputs a CSV with the four values: p808_mos, mos_sig, mos_bak, mos_ovr.

Supports two input modes:
  - folder: recursive glob of *.wav in a directory
  - wavscp: Kaldi-style wav.scp file (uttid wavpath per line)

Requires: pip install torchmetrics['audio']
"""

import argparse
import csv
import statistics
from pathlib import Path

import torch
import librosa

from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score


def load_wavscp(wav_scp_path):
    """Parse wav.scp into list of (uttid, wavpath) tuples."""
    pairs = []
    with open(wav_scp_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Compute DNSMOS for WAV files")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["folder", "wavscp"],
        default="folder",
        help="Input mode: folder (recursive *.wav) or wavscp (uttid wavpath per line)",
    )
    parser.add_argument(
        "wave_folder",
        type=str,
        nargs="?",
        default=None,
        help="Folder containing WAV files (required for mode=folder)",
    )
    parser.add_argument(
        "--wav_scp",
        type=str,
        default=None,
        help="Path to wav.scp file (required for mode=wavscp)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output CSV path (default: wave_folder/DNSMOS_results.csv or wav_scp dir)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate in Hz (default: 16000). Audio is resampled if needed.",
    )
    parser.add_argument(
        "--personalized",
        action="store_true",
        help="Whether interfering speaker is penalized",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (cpu, cuda, cuda:0, etc.). Default: auto",
    )
    args = parser.parse_args()

    if args.mode == "folder":
        if args.wave_folder is None:
            raise SystemExit("Error: wave_folder required for mode=folder")
        wave_folder = Path(args.wave_folder)
        if not wave_folder.exists() or not wave_folder.is_dir():
            raise SystemExit(f"Error: wave_folder '{wave_folder}' does not exist or is not a directory")
        def _file_display(p):
            try:
                return str(p.relative_to(wave_folder))
            except ValueError:
                return p.name
        wav_items = [(p.name, p, _file_display(p)) for p in sorted(wave_folder.rglob("*.wav"))]
        if not wav_items:
            raise SystemExit(f"Error: no .wav files found in '{wave_folder}'")
        default_output_dir = wave_folder
    else:
        if args.wav_scp is None:
            raise SystemExit("Error: --wav_scp required for mode=wavscp")
        wav_scp_path = Path(args.wav_scp)
        if not wav_scp_path.exists():
            raise SystemExit(f"Error: wav.scp '{wav_scp_path}' does not exist")
        pairs = load_wavscp(wav_scp_path)
        if not pairs:
            raise SystemExit(f"Error: no entries in wav.scp '{wav_scp_path}'")
        wav_items = [(uttid, Path(wavpath), wavpath) for uttid, wavpath in pairs]
        default_output_dir = wav_scp_path.parent

    output_path = (Path(args.output) / "DNSMOS_results.csv") if args.output else (default_output_dir / "DNSMOS_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Processing {len(wav_items)} WAV files ({args.mode} mode)")
    print(f"Output: {output_path}")
    print(f"Sample rate: {args.sample_rate} Hz, device: {device}")

    fieldnames = ["uttid", "file", "p808_mos", "mos_sig", "mos_bak", "mos_ovr"]
    with open(output_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    results = []
    written_count = 0
    for i, (uttid, wav_path, file_display) in enumerate(wav_items):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  [{i + 1}/{len(wav_items)}] {uttid}")

        try:
            audio, sr = librosa.load(str(wav_path), sr=args.sample_rate, mono=True)
        except Exception as e:
            print(f"  Warning: failed to load {wav_path}: {e}")
            continue

        # shape: (time,)
        preds = torch.from_numpy(audio).float().to(device)

        with torch.no_grad():
            dnsmos = deep_noise_suppression_mean_opinion_score(
                preds,
                fs=args.sample_rate,
                personalized=args.personalized,
                device=device,
            )

        values = dnsmos.cpu().tolist()
        if isinstance(values, float):
            values = [values]
        p808_mos, mos_sig, mos_bak, mos_ovr = values[:4]

        results.append({
            "uttid": uttid,
            "file": file_display,
            "p808_mos": p808_mos,
            "mos_sig": mos_sig,
            "mos_bak": mos_bak,
            "mos_ovr": mos_ovr,
        })

        # Write to file every 100 samples
        if len(results) % 100 == 0:
            with open(output_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerows(results[written_count:written_count + 100])
            written_count += 100
            print(f"  Flushed {written_count} rows to {output_path}")

    # Write any remaining results
    if written_count < len(results):
        with open(output_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerows(results[written_count:])

    print(f"Done. Wrote {len(results)} rows to {output_path}")

    # Print summary
    if results:
        p808 = [r["p808_mos"] for r in results]
        sig = [r["mos_sig"] for r in results]
        bak = [r["mos_bak"] for r in results]
        ovr = [r["mos_ovr"] for r in results]
        print(f"Summary: p808_mos={statistics.mean(p808):.3f}, mos_sig={statistics.mean(sig):.3f}, "
              f"mos_bak={statistics.mean(bak):.3f}, mos_ovr={statistics.mean(ovr):.3f}")


if __name__ == "__main__":
    main()
