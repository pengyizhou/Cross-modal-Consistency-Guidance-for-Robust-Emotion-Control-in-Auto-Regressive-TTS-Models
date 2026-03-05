#!/usr/bin/env python3
'''
Emotion recognition using emotion2vec via FunASR.

9-class emotions (emotion2vec_plus_large):
    0: angry, 1: disgusted, 2: fearful, 3: happy,
    4: neutral, 5: other, 6: sad, 7: surprised, 8: unknown
'''

import argparse
from funasr import AutoModel
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Emotion recognition using emotion2vec model')
    parser.add_argument('--wav_scp', type=str, required=True,
                        help='Path to wav.scp file (Kaldi format: utt_id path)')
    parser.add_argument('--output_dir', type=str, default="./outputs",
                        help='Directory to save output results')
    parser.add_argument('--model_id', type=str, default="iic/emotion2vec_plus_large",
                        help='Model ID for emotion2vec (auto-downloaded from ModelScope)')
    parser.add_argument('--extract_embedding', action='store_true',
                        help='Whether to extract embeddings')
    return parser.parse_args()

args = parse_args()

print(f"Using model: {args.model_id}")
print(f"Processing wav_scp: {args.wav_scp}")
print(f"Output directory: {args.output_dir}")

os.makedirs(args.output_dir, exist_ok=True)

model = AutoModel(
    model=args.model_id,
    hub="ms",
)

rec_result = model.generate(args.wav_scp, output_dir=args.output_dir,
                            granularity="utterance",
                            extract_embedding=args.extract_embedding,
                            batch_size=64)

output_file = f"{args.output_dir}/emo_results.txt"
print(f"Saving results to: {output_file}")
print(f"Processing {len(rec_result)} audio samples...")

with open(output_file, "w") as f:
    for result in rec_result:
        max_score = max(result['scores'])
        max_label = result['labels'][result['scores'].index(max_score)]
        max_label = max_label.split('/')[1] if max_label != "<unk>" else max_label
        f.write(f"{result['key']} {max_label} {max_score}\n")

print(f"Done! Results saved to {output_file}")
