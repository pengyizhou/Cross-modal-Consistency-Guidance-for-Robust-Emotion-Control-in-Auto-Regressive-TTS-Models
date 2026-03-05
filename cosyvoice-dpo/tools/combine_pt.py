#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch


def load_dict(path: Path) -> dict:
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"{path} does not contain a dictionary (found {type(value).__name__})")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple dict .pt files into one.")
    parser.add_argument("output", type=Path, help="destination .pt file")
    parser.add_argument("inputs", type=Path, nargs="+", help="source .pt files (dicts)")
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="permit later files to overwrite duplicate keys from earlier ones",
    )
    args = parser.parse_args()

    merged = {}

    # Load following inputs first (so first input's duplicates get ext-)
    for src in args.inputs[1:]:
        data = load_dict(src)
        for key, value in data.items():
            if key in merged and not args.allow_overwrite:
                print(f"duplicate: {key} skipped (from {src})", file=sys.stderr)
                continue
            merged[key] = value

    # Load first input last; duplicates get ext- prefix
    if args.inputs:
        data = load_dict(args.inputs[0])
        for key, value in data.items():
            out_key = key
            while out_key in merged:
                if args.allow_overwrite:
                    break
                out_key = "ext-" + out_key
                print(f"duplicate: {key} -> {out_key} (from {args.inputs[0]})", file=sys.stderr)
            merged[out_key] = value

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)