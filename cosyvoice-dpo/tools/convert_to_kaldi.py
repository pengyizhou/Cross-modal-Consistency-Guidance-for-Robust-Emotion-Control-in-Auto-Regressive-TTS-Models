#!/usr/bin/env python3
"""
Convert TextrolSpeech JSONL files to Kaldi format.
This script processes random_train.jsonl and random_test.jsonl files and creates
Kaldi format files in ./data/TextrolSpeech/train and ./data/TextrolSpeech/test directories.
"""

import json
import os
import argparse
from pathlib import Path


def create_kaldi_files(jsonl_file, output_dir):
    """
    Create Kaldi format files from a JSONL file.
    
    Args:
        jsonl_file (str): Path to the input JSONL file
        output_dir (str): Output directory for Kaldi files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize lists to store data
    wav_scp = []  # utterance_id /path/to/audio.wav
    text = []     # utterance_id transcript
    utt2spk = []  # utterance_id speaker_id
    utt2gender = []  # utterance_id gender
    utt2emotion = []  # utterance_id emotion
    utt2pitch = []   # utterance_id pitch
    utt2energy = []  # utterance_id energy
    style_prompts = []  # utterance_id style_prompt
    
    print(f"Processing {jsonl_file}...")
    
    # Process each line in the JSONL file
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 10000 == 0:
                print(f"  Processed {line_num} lines...")
                
            try:
                data = json.loads(line.strip())
                
                # Extract fields
                utterance_id = data['ID']
                # Clean utterance ID by replacing spaces and brackets with underscores
                utterance_id = utterance_id.replace(' ', '_').replace('(', '_').replace(')', '_').replace('[', '_').replace(']', '_')
                audio_path = data['audio_path']
                # gender = data['gender']
                # pitch = data['pitch']
                # energy = data['energy']
                # emotion = data['emotion']
                style_prompt = data['style_prompt']
                content_prompt = data['content_prompt']
                file_exists = data.get('file_exists', True)
                
                # Skip if file doesn't exist
                if not file_exists:
                    continue
                
                # Create speaker ID from the utterance ID
                # Extract dataset and speaker info from the ID
                speaker_id = utterance_id.split('-')[0] if '-' in utterance_id else utterance_id.split('_')[0]
                
                # Prepare entries
                wav_scp.append(f"{utterance_id} {audio_path}")
                text.append(f"{utterance_id} {content_prompt}")
                # utt2spk.append(f"{utterance_id} {speaker_id}")
                # utt2gender.append(f"{utterance_id} {gender}")
                # utt2emotion.append(f"{utterance_id} {emotion}")
                # utt2pitch.append(f"{utterance_id} {pitch}")
                # utt2energy.append(f"{utterance_id} {energy}")
                style_prompts.append(f"{utterance_id} {style_prompt}")
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except KeyError as e:
                print(f"Missing key in line {line_num}: {e}")
                continue
    
    print(f"  Processed {len(wav_scp)} valid entries")
    
    # Sort all lists by utterance ID for consistency
    wav_scp.sort()
    text.sort()
    utt2spk.sort()
    utt2gender.sort()
    utt2emotion.sort()
    utt2pitch.sort()
    utt2energy.sort()
    style_prompts.sort()
    
    # Write Kaldi format files
    files_to_write = [
        ('wav.scp', wav_scp),
        ('text', text),
        # ('utt2spk', utt2spk),
        # ('utt2gender', utt2gender),
        # ('utt2emotion', utt2emotion),
        # ('utt2pitch', utt2pitch),
        # ('utt2energy', utt2energy),
        ('style_prompts', style_prompts)
    ]
    
    for filename, data_list in files_to_write:
        file_path = os.path.join(output_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in data_list:
                f.write(line + '\n')
        print(f"  Created {file_path} with {len(data_list)} entries")
    
    # Create spk2utt from utt2spk
    spk2utt = {}
    for line in utt2spk:
        utt_id, spk_id = line.split(' ', 1)
        if spk_id not in spk2utt:
            spk2utt[spk_id] = []
        spk2utt[spk_id].append(utt_id)
    
    # Write spk2utt
    spk2utt_path = os.path.join(output_dir, 'spk2utt')
    with open(spk2utt_path, 'w', encoding='utf-8') as f:
        for spk_id in sorted(spk2utt.keys()):
            utts = ' '.join(sorted(spk2utt[spk_id]))
            f.write(f"{spk_id} {utts}\n")
    print(f"  Created {spk2utt_path} with {len(spk2utt)} speakers")


def main():
    parser = argparse.ArgumentParser(description='Convert TextrolSpeech JSONL files to Kaldi format')
    parser.add_argument('--train_jsonl', 
                      required=False,
                      help='Path to train JSONL file')
    parser.add_argument('--test_jsonl', 
                        required=True,
                        help='Path to test JSONL file')
    parser.add_argument('--output_dir', 
                      required=True,
                      help='Output directory for Kaldi files')
    
    args = parser.parse_args()
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process train set
    if args.train_jsonl is not None:
        if os.path.exists(args.train_jsonl):
            train_output_dir = os.path.join(args.output_dir, 'train')
            create_kaldi_files(args.train_jsonl, train_output_dir)
        else:
            print(f"Train file not found: {args.train_jsonl}")
        
    # Process test set
    if os.path.exists(args.test_jsonl):
        test_output_dir = os.path.join(args.output_dir, 'test')
        create_kaldi_files(args.test_jsonl, test_output_dir)
    else:
        print(f"Test file not found: {args.test_jsonl}")
    
    print("\nConversion completed!")
    print(f"Kaldi format files created in: {args.output_dir}")
    print("\nGenerated files for each set:")
    print("  - wav.scp: utterance_id to audio file mapping")
    print("  - text: utterance_id to transcript mapping")
    print("  - utt2spk: utterance to speaker mapping")
    print("  - spk2utt: speaker to utterances mapping")
    print("  - utt2gender: utterance to gender mapping")
    print("  - utt2emotion: utterance to emotion mapping")
    print("  - utt2pitch: utterance to pitch mapping")
    print("  - utt2energy: utterance to energy mapping")
    print("  - style_prompts: utterance to style prompt mapping")


if __name__ == "__main__":
    main()
