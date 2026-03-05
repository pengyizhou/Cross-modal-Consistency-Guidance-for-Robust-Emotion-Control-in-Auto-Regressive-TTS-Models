#!/usr/bin/env python3
"""
Convert emotion JSONL files to Kaldi format.
Processes JSONL files with emotion/reference audio fields and creates
Kaldi format files (wav.scp, text, utt2spk, utt2emotion, etc.) for evaluation.
"""

import json
import os
import argparse


def create_kaldi_files(jsonl_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    wav_scp = []
    text = []
    utt2spk = []
    utt2emotion = []
    ref_wav_scp = []
    ref_text = []
    utt2ref_emotion = []

    print(f"Processing {jsonl_file}...")

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if line_num % 10000 == 0:
                print(f"  Processed {line_num} lines...")

            try:
                data = json.loads(line)
                utterance_id = data['ID']
                utterance_id = utterance_id.replace(' ', '_').replace('(', '_').replace(')', '_').replace('[', '_').replace(']', '_')
                audio_path = data['audio_path']
                emotion = data['emotion']
                content = data['content']
                ref_audio_path = data['ref_audio_path']
                ref_emotion = data['ref_emotion']
                ref_content = data['ref_content']

                speaker_id = utterance_id.split('-')[0] if '-' in utterance_id else utterance_id.split('_')[0]

                wav_scp.append(f"{utterance_id} {audio_path}")
                text.append(f"{utterance_id} {content}")
                utt2spk.append(f"{utterance_id} {speaker_id}")
                utt2emotion.append(f"{utterance_id} {emotion}")
                ref_wav_scp.append(f"{utterance_id} {ref_audio_path}")
                ref_text.append(f"{utterance_id} {ref_content}")
                utt2ref_emotion.append(f"{utterance_id} {ref_emotion}")

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except KeyError as e:
                print(f"Missing key in line {line_num}: {e}")
                continue

    print(f"  Processed {len(wav_scp)} valid entries")

    wav_scp.sort()
    text.sort()
    utt2spk.sort()
    utt2emotion.sort()
    ref_wav_scp.sort()
    ref_text.sort()
    utt2ref_emotion.sort()

    files_to_write = [
        ('wav.scp', wav_scp),
        ('text', text),
        ('utt2spk', utt2spk),
        ('utt2emotion', utt2emotion),
        ('ref_wav.scp', ref_wav_scp),
        ('ref_text', ref_text),
        ('utt2ref_emotion', utt2ref_emotion),
    ]

    for filename, data_list in files_to_write:
        file_path = os.path.join(output_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in data_list:
                f.write(line + '\n')
        print(f"  Created {file_path} with {len(data_list)} entries")

    spk2utt = {}
    for line in utt2spk:
        utt_id, spk_id = line.split(' ', 1)
        if spk_id not in spk2utt:
            spk2utt[spk_id] = []
        spk2utt[spk_id].append(utt_id)

    spk2utt_path = os.path.join(output_dir, 'spk2utt')
    with open(spk2utt_path, 'w', encoding='utf-8') as f:
        for spk_id in sorted(spk2utt.keys()):
            utts = ' '.join(sorted(spk2utt[spk_id]))
            f.write(f"{spk_id} {utts}\n")
    print(f"  Created {spk2utt_path} with {len(spk2utt)} speakers")


def main():
    parser = argparse.ArgumentParser(description='Convert emotion JSONL files to Kaldi format')
    parser.add_argument('--train_jsonl', required=False, help='Path to train JSONL file')
    parser.add_argument('--test_jsonl', required=True, help='Path to test JSONL file')
    parser.add_argument('--output_dir', required=True, help='Output directory for Kaldi files')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.train_jsonl is not None:
        if os.path.exists(args.train_jsonl):
            train_output_dir = os.path.join(args.output_dir, 'train')
            create_kaldi_files(args.train_jsonl, train_output_dir)
        else:
            print(f"Train file not found: {args.train_jsonl}")

    if os.path.exists(args.test_jsonl):
        test_output_dir = os.path.join(args.output_dir, 'test')
        create_kaldi_files(args.test_jsonl, test_output_dir)
    else:
        print(f"Test file not found: {args.test_jsonl}")

    print(f"\nKaldi format files created in: {args.output_dir}")


if __name__ == "__main__":
    main()
